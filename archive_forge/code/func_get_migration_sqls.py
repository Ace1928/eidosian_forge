from __future__ import annotations
import abc
from pathlib import Path
from typing import Optional, Any, Dict, List, Union, Type, TYPE_CHECKING
def get_migration_sqls(self, include: Optional[List[str]]=None, exclude: Optional[List[str]]=None, path: Optional[Union[str, Path]]=None) -> List[str]:
    """
        Returns the migration sqls
        """
    sqls = []
    path = path or 'migrations'
    if isinstance(path, str):
        path = self.sql_path.joinpath(path)
    if not path.exists():
        return sqls
    for name in path.iterdir():
        if name.suffix != '.sql':
            continue
        name_stem = name.stem.split('.', 1)[0]
        if include and name_stem not in include:
            continue
        if exclude and name_stem in exclude:
            continue
        sqls.append(name_stem)
    return sqls