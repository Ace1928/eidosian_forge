import os
from pathlib import Path
import sys
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import iniconfig
from .exceptions import UsageError
from _pytest.outcomes import fail
from _pytest.pathlib import absolutepath
from _pytest.pathlib import commonpath
from _pytest.pathlib import safe_exists
def load_config_dict_from_file(filepath: Path) -> Optional[Dict[str, Union[str, List[str]]]]:
    """Load pytest configuration from the given file path, if supported.

    Return None if the file does not contain valid pytest configuration.
    """
    if filepath.suffix == '.ini':
        iniconfig = _parse_ini_config(filepath)
        if 'pytest' in iniconfig:
            return dict(iniconfig['pytest'].items())
        elif filepath.name == 'pytest.ini':
            return {}
    elif filepath.suffix == '.cfg':
        iniconfig = _parse_ini_config(filepath)
        if 'tool:pytest' in iniconfig.sections:
            return dict(iniconfig['tool:pytest'].items())
        elif 'pytest' in iniconfig.sections:
            fail(CFG_PYTEST_SECTION.format(filename='setup.cfg'), pytrace=False)
    elif filepath.suffix == '.toml':
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib
        toml_text = filepath.read_text(encoding='utf-8')
        try:
            config = tomllib.loads(toml_text)
        except tomllib.TOMLDecodeError as exc:
            raise UsageError(f'{filepath}: {exc}') from exc
        result = config.get('tool', {}).get('pytest', {}).get('ini_options', None)
        if result is not None:

            def make_scalar(v: object) -> Union[str, List[str]]:
                return v if isinstance(v, list) else str(v)
            return {k: make_scalar(v) for k, v in result.items()}
    return None