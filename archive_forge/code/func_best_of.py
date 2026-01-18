from typing import List, Iterable
import cirq
@staticmethod
def best_of(lines: Iterable[LineSequence], length: int) -> 'GridQubitLineTuple':
    lines = list(lines)
    longest = max(lines, key=len) if lines else []
    if len(longest) < length:
        raise NotFoundError('No line placement with desired length found.')
    return GridQubitLineTuple(longest[:length])