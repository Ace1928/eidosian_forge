import os
from pathlib import Path
from typing import Optional, Union
def as_import_path(file: Union[Path, str], *, suffix: Optional[str]=None, relative_to: Path=PACKAGE_DIR) -> str:
    """Path of the file as a LangChain import exclude langchain top namespace."""
    if isinstance(file, str):
        file = Path(file)
    path = get_relative_path(file, relative_to=relative_to)
    if file.is_file():
        path = path[:-len(file.suffix)]
    import_path = path.replace(SEPARATOR, '.')
    if suffix:
        import_path += '.' + suffix
    return import_path