import sys
from typing import Any, Callable, Dict
from .version import version_short
def getattr_migration(module: str) -> Callable[[str], Any]:
    """Implement PEP 562 for objects that were either moved or removed on the migration
    to V2.

    Args:
        module: The module name.

    Returns:
        A callable that will raise an error if the object is not found.
    """
    from .errors import PydanticImportError

    def wrapper(name: str) -> object:
        """Raise an error if the object is not found, or warn if it was moved.

        In case it was moved, it still returns the object.

        Args:
            name: The object name.

        Returns:
            The object.
        """
        if name == '__path__':
            raise AttributeError(f'module {module!r} has no attribute {name!r}')
        import warnings
        from ._internal._validators import import_string
        import_path = f'{module}:{name}'
        if import_path in MOVED_IN_V2.keys():
            new_location = MOVED_IN_V2[import_path]
            warnings.warn(f'`{import_path}` has been moved to `{new_location}`.')
            return import_string(MOVED_IN_V2[import_path])
        if import_path in DEPRECATED_MOVED_IN_V2:
            return import_string(DEPRECATED_MOVED_IN_V2[import_path])
        if import_path in REDIRECT_TO_V1:
            new_location = REDIRECT_TO_V1[import_path]
            warnings.warn(f'`{import_path}` has been removed. We are importing from `{new_location}` instead.See the migration guide for more details: https://docs.pydantic.dev/latest/migration/')
            return import_string(REDIRECT_TO_V1[import_path])
        if import_path == 'pydantic:BaseSettings':
            raise PydanticImportError(f'`BaseSettings` has been moved to the `pydantic-settings` package. See https://docs.pydantic.dev/{version_short()}/migration/#basesettings-has-moved-to-pydantic-settings for more details.')
        if import_path in REMOVED_IN_V2:
            raise PydanticImportError(f'`{import_path}` has been removed in V2.')
        globals: Dict[str, Any] = sys.modules[module].__dict__
        if name in globals:
            return globals[name]
        raise AttributeError(f'module {module!r} has no attribute {name!r}')
    return wrapper