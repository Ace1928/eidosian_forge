import argparse
import operator
import sys
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union
import gitlab
import gitlab.base
import gitlab.v4.objects
from gitlab import cli
from gitlab.exceptions import GitlabCiLintError
def do_project_export_download(self) -> None:
    try:
        project = self.gl.projects.get(self.parent_args['project_id'], lazy=True)
        export_status = project.exports.get()
        if TYPE_CHECKING:
            assert export_status is not None
        data = export_status.download()
        if TYPE_CHECKING:
            assert data is not None
            assert isinstance(data, bytes)
        sys.stdout.buffer.write(data)
    except Exception as e:
        cli.die('Impossible to download the export', e)