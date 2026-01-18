from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
def get_platonic_tables(ManifoldTable):

    class PlatonicManifoldTable(ManifoldTable):
        """
        Iterator for platonic hyperbolic manifolds.
        """

        def __init__(self, table='', db_path=platonic_database_path, **filter_args):
            ManifoldTable.__init__(self, table=table, db_path=db_path, **filter_args)

        def _configure(self, **kwargs):
            ManifoldTable._configure(self, **kwargs)
            conditions = []
            if 'solids' in kwargs:
                N = int(kwargs['solids'])
                conditions.append('solids = %d' % N)
            if self._filter:
                if len(conditions) > 0:
                    self._filter += ' and ' + ' and '.join(conditions)
            else:
                self._filter = ' and '.join(conditions)

    class TetrahedralOrientableCuspedCensus(PlatonicManifoldTable):
        """
        Iterator for the tetrahedral orientable cusped hyperbolic manifolds up to
        25 tetrahedra, i.e., manifolds that admit a tessellation by regular ideal
        hyperbolic tetrahedra.

        >>> for M in TetrahedralOrientableCuspedCensus(solids = 5): # doctest: +NUMERIC6
        ...     print(M, M.volume())
        otet05_00000(0,0) 5.07470803
        otet05_00001(0,0)(0,0) 5.07470803
        >>> TetrahedralOrientableCuspedCensus.identify(Manifold("m004"))
        otet02_00001(0,0)


        """
        _regex = re.compile('otet\\d+_\\d+')

        def __init__(self, **kwargs):
            return PlatonicManifoldTable.__init__(self, table='tetrahedral_orientable_cusped_census', **kwargs)

    class TetrahedralNonorientableCuspedCensus(PlatonicManifoldTable):
        """
        Iterator for the tetrahedral non-orientable cusped hyperbolic manifolds up to
        21 tetrahedra, i.e., manifolds that admit a tessellation by regular ideal
        hyperbolic tetrahedra.

        >>> len(TetrahedralNonorientableCuspedCensus)
        25194
        >>> list(TetrahedralNonorientableCuspedCensus[:1.3])
        [ntet01_00000(0,0)]

        """
        _regex = re.compile('ntet\\d+_\\d+')

        def __init__(self, **kwargs):
            return PlatonicManifoldTable.__init__(self, 'tetrahedral_nonorientable_cusped_census', **kwargs)

    class OctahedralOrientableCuspedCensus(PlatonicManifoldTable):
        """
        Iterator for the octahedral orientable cusped hyperbolic manifolds up to
        7 octahedra, i.e., manifolds that admit a tessellation by regular ideal
        hyperbolic octahedra.

            >>> OctahedralOrientableCuspedCensus.identify(Manifold("5^2_1"))
            ooct01_00001(0,0)(0,0)

        For octahedral manifolds that are also the complement of an `Augmented
        Knotted Trivalent Graph (AugKTG) <http://arxiv.org/abs/0805.0094>`_, the
        corresponding link is included::

            >>> M = OctahedralOrientableCuspedCensus['ooct04_00034']
            >>> M.link()
            <Link: 4 comp; 17 cross>

        The link can be viewed with ``M.plink()``. To only see complements of
        AugKTGs, supply ``isAugKTG = True``::

         >>> len(OctahedralOrientableCuspedCensus(isAugKTG = True))
         238
         >>> for M in OctahedralOrientableCuspedCensus(isAugKTG = True)[:5]:
         ...     print(M, M.link().DT_code(DT_alpha=True))
         ooct02_00001(0,0)(0,0)(0,0)(0,0) DT[mdbceceJamHBlCKgdfI]
         ooct02_00002(0,0)(0,0)(0,0) DT[lcgbcIkhLBJecGaFD]
         ooct02_00003(0,0)(0,0)(0,0) DT[icebbGIAfhcEdB]
         ooct02_00005(0,0)(0,0)(0,0) DT[hcdbbFHegbDAc]
         ooct04_00027(0,0)(0,0)(0,0)(0,0) DT[zdpecbBujVtiWzsLQpxYREadhOKCmFgN]


        """
        _select = 'select name, triangulation, DT from %s '
        _regex = re.compile('ooct\\d+_\\d+')

        def __init__(self, **kwargs):
            return PlatonicManifoldTable.__init__(self, 'octahedral_orientable_cusped_census', **kwargs)

        def _configure(self, **kwargs):
            PlatonicManifoldTable._configure(self, **kwargs)
            conditions = []
            if 'isAugKTG' in kwargs:
                if kwargs['isAugKTG']:
                    conditions.append('isAugKTG = 1')
                else:
                    conditions.append('isAugKTG = 0')
            if self._filter:
                if len(conditions) > 0:
                    self._filter += ' and ' + ' and '.join(conditions)
            else:
                self._filter = ' and '.join(conditions)

        def _finalize(self, M, row):
            PlatonicManifoldTable._finalize(self, M, row)
            if row[2] and (not row[2] == 'Null'):
                M._set_DTcode(row[2])

    class OctahedralNonorientableCuspedCensus(PlatonicManifoldTable):
        """
        Iterator for the octahedral non-orientable cusped hyperbolic manifolds up to
        5 octahedra, i.e., manifolds that admit a tessellation by regular ideal
        hyperbolic octahedra.

        >>> for M in OctahedralNonorientableCuspedCensus(solids = 3, betti = 3,cusps = 4):
        ...     print(M, M.homology())
        noct03_00007(0,0)(0,0)(0,0)(0,0) Z/2 + Z + Z + Z
        noct03_00029(0,0)(0,0)(0,0)(0,0) Z/2 + Z + Z + Z
        noct03_00047(0,0)(0,0)(0,0)(0,0) Z/2 + Z + Z + Z
        noct03_00048(0,0)(0,0)(0,0)(0,0) Z/2 + Z + Z + Z

        """
        _regex = re.compile('noct\\d+_\\d+')

        def __init__(self, **kwargs):
            return PlatonicManifoldTable.__init__(self, 'octahedral_nonorientable_cusped_census', **kwargs)

    class CubicalOrientableCuspedCensus(PlatonicManifoldTable):
        """
        Iterator for the cubical orientable cusped hyperbolic manifolds up to
        7 cubes, i.e., manifolds that admit a tessellation by regular ideal
        hyperbolic octahedra.

        >>> M = TetrahedralOrientableCuspedCensus['otet05_00001']
        >>> CubicalOrientableCuspedCensus.identify(M)
        ocube01_00002(0,0)(0,0)

        """
        _regex = re.compile('ocube\\d+_\\d+')

        def __init__(self, **kwargs):
            return PlatonicManifoldTable.__init__(self, 'cubical_orientable_cusped_census', **kwargs)

    class CubicalNonorientableCuspedCensus(PlatonicManifoldTable):
        """
        Iterator for the cubical non-orientable cusped hyperbolic manifolds up to
        5 cubes, i.e., manifolds that admit a tessellation by regular ideal
        hyperbolic octahedra.

        >>> for M in CubicalNonorientableCuspedCensus[-3:]: # doctest: +NUMERIC6
        ...     print(M, M.volume())
        ncube05_30945(0,0) 25.37354016
        ncube05_30946(0,0)(0,0) 25.37354016
        ncube05_30947(0,0)(0,0) 25.37354016

        """
        _regex = re.compile('ncube\\d+_\\d+')

        def __init__(self, **kwargs):
            return PlatonicManifoldTable.__init__(self, 'cubical_nonorientable_cusped_census', **kwargs)

    class DodecahedralOrientableCuspedCensus(PlatonicManifoldTable):
        """
        Iterator for the dodecahedral orientable cusped hyperbolic manifolds up to
        2 dodecahedra, i.e., manifolds that admit a tessellation by regular ideal
        hyperbolic dodecahedra.

        Complement of one of the dodecahedral knots by Aitchison and Rubinstein::

          >>> M=DodecahedralOrientableCuspedCensus['odode02_00913']
          >>> M.dehn_fill((1,0))
          >>> M.fundamental_group()
          Generators:
          <BLANKLINE>
          Relators:
          <BLANKLINE>

        """
        _regex = re.compile('odode\\d+_\\d+')

        def __init__(self, **kwargs):
            return PlatonicManifoldTable.__init__(self, 'dodecahedral_orientable_cusped_census', **kwargs)

    class DodecahedralNonorientableCuspedCensus(PlatonicManifoldTable):
        """
        Iterator for the dodecahedral non-orientable cusped hyperbolic manifolds up to
        2 dodecahedra, i.e., manifolds that admit a tessellation by regular ideal
        hyperbolic dodecahedra.

        >>> len(DodecahedralNonorientableCuspedCensus)
        4146

        """
        _regex = re.compile('ndode\\d+_\\d+')

        def __init__(self, **kwargs):
            return PlatonicManifoldTable.__init__(self, 'dodecahedral_nonorientable_cusped_census', **kwargs)

    class IcosahedralNonorientableClosedCensus(PlatonicManifoldTable):
        """
        Iterator for the icosahedral non-orientable closed hyperbolic manifolds up
        to 3 icosahedra, i.e., manifolds that admit a tessellation by regular finite
        hyperbolic icosahedra.

        >>> list(IcosahedralNonorientableClosedCensus)
        [nicocld02_00000(1,0)]

        """
        _regex = re.compile('nicocld\\d+_\\d+')

        def __init__(self, **kwargs):
            return PlatonicManifoldTable.__init__(self, 'icosahedral_nonorientable_closed_census', **kwargs)

    class IcosahedralOrientableClosedCensus(PlatonicManifoldTable):
        """
        Iterator for the icosahedral orientable closed hyperbolic manifolds up
        to 4 icosahedra, i.e., manifolds that admit a tessellation by regula finite
        hyperbolic icosahedra.

        >>> M = IcosahedralOrientableClosedCensus[0]
        >>> M.volume() # doctest: +NUMERIC6
        4.68603427
        >>> M
        oicocld01_00000(1,0)
        """
        _regex = re.compile('oicocld\\d+_\\d+')

        def __init__(self, **kwargs):
            return PlatonicManifoldTable.__init__(self, 'icosahedral_orientable_closed_census', **kwargs)

    class CubicalNonorientableClosedCensus(PlatonicManifoldTable):
        """
        Iterator for the cubical non-orientable closed hyperbolic manifolds up
        to 10 cubes, i.e., manifolds that admit a tessellation by regular finite
        hyperbolic cubes.

        >>> len(CubicalNonorientableClosedCensus)
        93

        """
        _regex = re.compile('ncube\\d+_\\d+')

        def __init__(self, **kwargs):
            return PlatonicManifoldTable.__init__(self, 'cubical_nonorientable_closed_census', **kwargs)

    class CubicalOrientableClosedCensus(PlatonicManifoldTable):
        """
        Iterator for the cubical orientable closed hyperbolic manifolds up
        to 10 cubes, i.e., manifolds that admit a tessellation by regular finite
        hyperbolic cubes.

        >>> len(CubicalOrientableClosedCensus)
        69

        """
        _regex = re.compile('ocube\\d+_\\d+')

        def __init__(self, **kwargs):
            return PlatonicManifoldTable.__init__(self, 'cubical_orientable_closed_census', **kwargs)

    class DodecahedralNonorientableClosedCensus(PlatonicManifoldTable):
        """
        Iterator for the dodecahedral non-orientable closed hyperbolic manifolds up
        to 2 dodecahedra, i.e., manifolds that admit a tessellation by regular finite
        hyperbolic dodecahedra with a dihedral angle of 72 degrees.

        >>> DodecahedralNonorientableClosedCensus[0].volume() # doctest: +NUMERIC6
        22.39812948

        """
        _regex = re.compile('ndodecld\\d+_\\d+')

        def __init__(self, **kwargs):
            return PlatonicManifoldTable.__init__(self, 'dodecahedral_nonorientable_closed_census', **kwargs)

    class DodecahedralOrientableClosedCensus(PlatonicManifoldTable):
        """
        Iterator for the dodecahedral orientable closed hyperbolic manifolds up
        to 3 dodecahedra, i.e., manifolds that admit a tessellation by regular finite
        hyperbolic dodecahedra with a dihedral angle of 72 degrees.

        The Seifert-Weber space::

          >>> M = DodecahedralOrientableClosedCensus(solids = 1)[-1]
          >>> M.homology()
          Z/5 + Z/5 + Z/5

        """
        _regex = re.compile('ododecld\\d+_\\d+')

        def __init__(self, **kwargs):
            return PlatonicManifoldTable.__init__(self, 'dodecahedral_orientable_closed_census', **kwargs)
    return [TetrahedralOrientableCuspedCensus(), TetrahedralNonorientableCuspedCensus(), OctahedralOrientableCuspedCensus(), OctahedralNonorientableCuspedCensus(), CubicalOrientableCuspedCensus(), CubicalNonorientableCuspedCensus(), DodecahedralOrientableCuspedCensus(), DodecahedralNonorientableCuspedCensus(), IcosahedralNonorientableClosedCensus(), IcosahedralOrientableClosedCensus(), CubicalNonorientableClosedCensus(), CubicalOrientableClosedCensus(), DodecahedralNonorientableClosedCensus(), DodecahedralOrientableClosedCensus()]