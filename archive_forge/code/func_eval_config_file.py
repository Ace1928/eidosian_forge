import re
import traceback
import types
from collections import OrderedDict
from os import getenv, path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generator, Iterator, List, NamedTuple,
from sphinx.errors import ConfigError, ExtensionError
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.i18n import format_date
from sphinx.util.osutil import cd, fs_encoding
from sphinx.util.tags import Tags
from sphinx.util.typing import NoneType
def eval_config_file(filename: str, tags: Optional[Tags]) -> Dict[str, Any]:
    """Evaluate a config file."""
    namespace: Dict[str, Any] = {}
    namespace['__file__'] = filename
    namespace['tags'] = tags
    with cd(path.dirname(filename)):
        try:
            with open(filename, 'rb') as f:
                code = compile(f.read(), filename.encode(fs_encoding), 'exec')
                exec(code, namespace)
        except SyntaxError as err:
            msg = __('There is a syntax error in your configuration file: %s\n')
            raise ConfigError(msg % err) from err
        except SystemExit as exc:
            msg = __('The configuration file (or one of the modules it imports) called sys.exit()')
            raise ConfigError(msg) from exc
        except ConfigError:
            raise
        except Exception as exc:
            msg = __('There is a programmable error in your configuration file:\n\n%s')
            raise ConfigError(msg % traceback.format_exc()) from exc
    return namespace