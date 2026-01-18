from itertools import islice
import re
import warnings
from io import StringIO, UnsupportedOperation
import json
import numpy as np
import numbers
from ase.atoms import Atoms
from ase.calculators.calculator import all_properties, Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.spacegroup.spacegroup import Spacegroup
from ase.parallel import paropen
from ase.constraints import FixAtoms, FixCartesian
from ase.io.formats import index2range
from ase.utils import reader
@reader
def read_xyz(fileobj, index=-1, properties_parser=key_val_str_to_dict):
    """
    Read from a file in Extended XYZ format

    index is the frame to read, default is last frame (index=-1).
    properties_parser is the parse to use when converting the properties line
    to a dictionary, ``extxyz.key_val_str_to_dict`` is the default and can
    deal with most use cases, ``extxyz.key_val_str_to_dict_regex`` is slightly
    faster but has fewer features.

    Extended XYZ format is an enhanced version of the `basic XYZ format
    <http://en.wikipedia.org/wiki/XYZ_file_format>`_ that allows extra
    columns to be present in the file for additonal per-atom properties as
    well as standardising the format of the comment line to include the
    cell lattice and other per-frame parameters.

    It's easiest to describe the format with an example.  Here is a
    standard XYZ file containing a bulk cubic 8 atom silicon cell ::

        8
        Cubic bulk silicon cell
        Si          0.00000000      0.00000000      0.00000000
        Si        1.36000000      1.36000000      1.36000000
        Si        2.72000000      2.72000000      0.00000000
        Si        4.08000000      4.08000000      1.36000000
        Si        2.72000000      0.00000000      2.72000000
        Si        4.08000000      1.36000000      4.08000000
        Si        0.00000000      2.72000000      2.72000000
        Si        1.36000000      4.08000000      4.08000000

    The first line is the number of atoms, followed by a comment and
    then one line per atom, giving the element symbol and cartesian
    x y, and z coordinates in Angstroms.

    Here's the same configuration in extended XYZ format ::

        8
        Lattice="5.44 0.0 0.0 0.0 5.44 0.0 0.0 0.0 5.44" Properties=species:S:1:pos:R:3 Time=0.0
        Si        0.00000000      0.00000000      0.00000000
        Si        1.36000000      1.36000000      1.36000000
        Si        2.72000000      2.72000000      0.00000000
        Si        4.08000000      4.08000000      1.36000000
        Si        2.72000000      0.00000000      2.72000000
        Si        4.08000000      1.36000000      4.08000000
        Si        0.00000000      2.72000000      2.72000000
        Si        1.36000000      4.08000000      4.08000000

    In extended XYZ format, the comment line is replaced by a series of
    key/value pairs.  The keys should be strings and values can be
    integers, reals, logicals (denoted by `T` and `F` for true and false)
    or strings. Quotes are required if a value contains any spaces (like
    `Lattice` above).  There are two mandatory parameters that any
    extended XYZ: `Lattice` and `Properties`. Other parameters --
    e.g. `Time` in the example above --- can be added to the parameter line
    as needed.

    `Lattice` is a Cartesian 3x3 matrix representation of the cell
    vectors, with each vector stored as a column and the 9 values listed in
    Fortran column-major order, i.e. in the form ::

      Lattice="R1x R1y R1z R2x R2y R2z R3x R3y R3z"

    where `R1x R1y R1z` are the Cartesian x-, y- and z-components of the
    first lattice vector (:math:`\\mathbf{a}`), `R2x R2y R2z` those of the second
    lattice vector (:math:`\\mathbf{b}`) and `R3x R3y R3z` those of the
    third lattice vector (:math:`\\mathbf{c}`).

    The list of properties in the file is described by the `Properties`
    parameter, which should take the form of a series of colon separated
    triplets giving the name, format (`R` for real, `I` for integer) and
    number of columns of each property. For example::

      Properties="species:S:1:pos:R:3:vel:R:3:select:I:1"

    indicates the first column represents atomic species, the next three
    columns represent atomic positions, the next three velcoities, and the
    last is an single integer called `select`. With this property
    definition, the line ::

      Si        4.08000000      4.08000000      1.36000000   0.00000000      0.00000000      0.00000000       1

    would describe a silicon atom at position (4.08,4.08,1.36) with zero
    velocity and the `select` property set to 1.

    The property names `pos`, `Z`, `mass`, and `charge` map to ASE
    :attr:`ase.atoms.Atoms.arrays` entries named
    `positions`, `numbers`, `masses` and `charges` respectively.

    Additional key-value pairs in the comment line are parsed into the
    :attr:`ase.Atoms.atoms.info` dictionary, with the following conventions

     - Values can be quoted with `""`, `''`, `[]` or `{}` (the latter are
       included to ease command-line usage as the `{}` are not treated
       specially by the shell)
     - Quotes within keys or values can be escaped with `\\"`.
     - Keys with special names `stress` or `virial` are treated as 3x3 matrices
       in Fortran order, as for `Lattice` above.
     - Otherwise, values with multiple elements are treated as 1D arrays, first
       assuming integer format and falling back to float if conversion is
       unsuccessful.
     - A missing value defaults to `True`, e.g. the comment line
       `"cutoff=3.4 have_energy"` leads to
       `{'cutoff': 3.4, 'have_energy': True}` in `atoms.info`.
     - Value strings starting with `"_JSON"` are interpreted as JSON content;
       similarly, when writing, anything which does not match the criteria above
       is serialised as JSON.

    The extended XYZ format is also supported by the
    the `Ovito <http://www.ovito.org>`_ visualisation tool
    (from `v2.4 beta
    <http://www.ovito.org/index.php/component/content/article?id=25>`_
    onwards).
    """
    if not isinstance(index, int) and (not isinstance(index, slice)):
        raise TypeError('Index argument is neither slice nor integer!')
    last_frame = None
    if isinstance(index, int) and index >= 0:
        last_frame = index
    elif isinstance(index, slice):
        if index.stop is not None and index.stop >= 0:
            last_frame = index.stop
    try:
        fileobj.seek(0)
    except UnsupportedOperation:
        fileobj = StringIO(fileobj.read())
        fileobj.seek(0)
    frames = []
    while True:
        frame_pos = fileobj.tell()
        line = fileobj.readline()
        if line.strip() == '':
            break
        try:
            natoms = int(line)
        except ValueError as err:
            raise XYZError('ase.io.extxyz: Expected xyz header but got: {}'.format(err))
        fileobj.readline()
        for i in range(natoms):
            fileobj.readline()
        nvec = 0
        while True:
            lastPos = fileobj.tell()
            line = fileobj.readline()
            if line.lstrip().startswith('VEC'):
                nvec += 1
                if nvec > 3:
                    raise XYZError('ase.io.extxyz: More than 3 VECX entries')
            else:
                fileobj.seek(lastPos)
                break
        frames.append((frame_pos, natoms, nvec))
        if last_frame is not None and len(frames) > last_frame:
            break
    trbl = index2range(index, len(frames))
    for index in trbl:
        frame_pos, natoms, nvec = frames[index]
        fileobj.seek(frame_pos)
        assert int(fileobj.readline()) == natoms
        yield _read_xyz_frame(fileobj, natoms, properties_parser, nvec)