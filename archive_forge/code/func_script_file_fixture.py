import importlib.machinery
import os
import shutil
import textwrap
from sqlalchemy.testing import config
from sqlalchemy.testing import provision
from . import util as testing_util
from .. import command
from .. import script
from .. import util
from ..script import Script
from ..script import ScriptDirectory
from alembic import context
from alembic import op
from alembic import op
from alembic import op
from alembic import op
from alembic import op
from alembic import op
def script_file_fixture(txt):
    dir_ = os.path.join(_get_staging_directory(), 'scripts')
    path = os.path.join(dir_, 'script.py.mako')
    with open(path, 'w') as f:
        f.write(txt)