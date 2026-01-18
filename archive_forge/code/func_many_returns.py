import enum
import typing as T
@property
def many_returns(self) -> T.List[DocstringReturns]:
    """Return a list of information on function return."""
    return [item for item in self.meta if isinstance(item, DocstringReturns)]