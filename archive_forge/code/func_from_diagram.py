import abc
import functools
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set, TYPE_CHECKING, Union
from typing_extensions import Self
import numpy as np
from cirq import ops, protocols
@staticmethod
def from_diagram(diagram: str) -> List['GridQubit']:
    """Parse ASCII art into device layout info.

        As an example, the below diagram will create a list of
        GridQubit in a pyramid structure.

        ```
        ---A---
        --AAA--
        -AAAAA-
        AAAAAAA
        ```

        You can use any character other than a hyphen, period or space to mark
        a qubit. As an example, the qubits for a Bristlecone device could be
        represented by the below diagram. This produces a diamond-shaped grid
        of qids, and qids with the same letter correspond to the same readout
        line.

        ```
        .....AB.....
        ....ABCD....
        ...ABCDEF...
        ..ABCDEFGH..
        .ABCDEFGHIJ.
        ABCDEFGHIJKL
        .CDEFGHIJKL.
        ..EFGHIJKL..
        ...GHIJKL...
        ....IJKL....
        .....KL.....
        ```

        Args:
            diagram: String representing the qubit layout. Each line represents
                a row. Alphanumeric characters are assigned as qid.
                Dots ('.'), dashes ('-'), and spaces (' ') are treated as
                empty locations in the grid. If diagram has characters other
                than alphanumerics, spacers, and newlines ('\\n'), an error will
                be thrown. The top-left corner of the diagram will be have
                coordinate (0,0).

        Returns:
            A list of GridQubit corresponding to qubits in the provided diagram

        Raises:
            ValueError: If the input string contains an invalid character.
        """
    coords = _ascii_diagram_to_coords(diagram)
    return [GridQubit(*c) for c in coords]