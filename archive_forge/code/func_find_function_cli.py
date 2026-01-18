from typing import Optional, Tuple
from catalogue import RegistryError
from wasabi import msg
from ..util import registry
from ._util import Arg, Opt, app
@app.command('find-function')
def find_function_cli(func_name: str=Arg(..., help='Name of the registered function.'), registry_name: Optional[str]=Opt(None, '--registry', '-r', help='Name of the catalogue registry.')):
    """
    Find the module, path and line number to the file the registered
    function is defined in, if available.

    func_name (str): Name of the registered function.
    registry_name (Optional[str]): Name of the catalogue registry.

    DOCS: https://spacy.io/api/cli#find-function
    """
    if not registry_name:
        registry_names = registry.get_registry_names()
        for name in registry_names:
            if registry.has(name, func_name):
                registry_name = name
                break
    if not registry_name:
        msg.fail(f"Couldn't find registered function: '{func_name}'", exits=1)
    assert registry_name is not None
    find_function(func_name, registry_name)