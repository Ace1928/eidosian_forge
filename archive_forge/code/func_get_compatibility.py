import sys
from typing import Optional, Sequence
import requests
import typer
from wasabi import msg
from .. import about
from ..errors import OLD_MODEL_SHORTCUTS
from ..util import (
from ._util import SDIST_SUFFIX, WHEEL_SUFFIX, Arg, Opt, app
def get_compatibility() -> dict:
    if is_prerelease_version(about.__version__):
        version: Optional[str] = about.__version__
    else:
        version = get_minor_version(about.__version__)
    r = requests.get(about.__compatibility__)
    if r.status_code != 200:
        msg.fail(f'Server error ({r.status_code})', f"Couldn't fetch compatibility table. Please find a package for your spaCy installation (v{about.__version__}), and download it manually. For more details, see the documentation: https://spacy.io/usage/models", exits=1)
    comp_table = r.json()
    comp = comp_table['spacy']
    if version not in comp:
        msg.fail(f'No compatible packages found for v{version} of spaCy', exits=1)
    return comp[version]