import os
from typing import Callable, Generator, Union
def filtered_dir(root: str, include_fn: Union[Callable[[str, str], bool], Callable[[str], bool]], exclude_fn: Union[Callable[[str, str], bool], Callable[[str], bool]]) -> Generator[str, None, None]:
    """Simple generator to walk a directory."""
    import inspect

    def _include_fn(path: str, root: str) -> bool:
        return include_fn(path, root) if len(inspect.signature(include_fn).parameters) == 2 else include_fn(path)

    def _exclude_fn(path: str, root: str) -> bool:
        return exclude_fn(path, root) if len(inspect.signature(exclude_fn).parameters) == 2 else exclude_fn(path)
    for dirpath, _, files in os.walk(root):
        for fname in files:
            file_path = os.path.join(dirpath, fname)
            if _include_fn(file_path, root) and (not _exclude_fn(file_path, root)):
                yield file_path