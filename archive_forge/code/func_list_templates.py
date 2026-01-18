import pathlib
from typing import List
from ray.util.annotations import DeveloperAPI
@staticmethod
def list_templates() -> List[pathlib.Path]:
    """List the available HTML templates.

        Returns:
            A list of files with .html.j2 extensions inside ../templates/
        """
    return (pathlib.Path(__file__).parent / 'templates').glob('*.html.j2')