from typing import Iterator, List
def post_order_iter(self) -> Iterator['Operator']:
    """Depth-first traversal of this operator and its input dependencies."""
    for op in self.input_dependencies:
        yield from op.post_order_iter()
    yield self