import sys
import os
from typing import List, Tuple, Optional
from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter
from project import (QmlProjectData, check_qml_decorators, is_python_file,
def qmllint(self):
    """Run qmllint on .qml files."""
    self.build()
    for sub_project_file in self.project.sub_projects_files:
        Project(project_file=sub_project_file)._qmllint()
    self._qmllint()