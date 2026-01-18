import typing
import functools
import pathlib
import importlib.util
from lazyops.utils.logs import default_logger as logger
from lazyops.utils.lazylib import LazyLib
from lazyops.utils.helpers import is_coro_func
def resolve_missing_custom(modules: typing.Union[str, typing.List], packages: typing.Union[str, typing.List]=None, required: bool=True):
    """
    Handles custom use cases like `torch` where we need to
    have a extra index to install from
    """
    if not isinstance(modules, list):
        modules = [modules]
    if packages is not None and (not isinstance(packages, list)):
        packages = [packages]
    elif packages is None:
        packages = modules
    module_names = [module.split(' ', 1)[0] for module in modules]
    kind = 'required' if required else 'optionally required'
    logger.info(f'{', '.join(module_names)} are {kind}. Installing...')
    for module, pkg in zip(modules, packages):
        module_name = LazyLib.get_requirement(module, True)
        if LazyLib.is_available(module_name):
            continue
        LazyLib.install_library(pkg)