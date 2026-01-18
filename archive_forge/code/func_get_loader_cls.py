from celery.utils.imports import import_from_cwd, symbol_by_name
def get_loader_cls(loader):
    """Get loader class by name/alias."""
    return symbol_by_name(loader, LOADER_ALIASES, imp=import_from_cwd)