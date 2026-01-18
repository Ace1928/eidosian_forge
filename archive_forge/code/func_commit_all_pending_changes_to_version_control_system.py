import re
import ast
import logging
from typing import List, Dict, Any, Union
import numpy as np
import logging
from typing import List
import os
import logging
from typing import Dict, List, Union
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Union
import json
import xml.etree.ElementTree as ET
import logging
import os
import subprocess
import logging
from typing import List
import ast
import logging
from typing import List, Dict
import logging
from typing import Dict
import ast
import networkx as nx
import matplotlib.pyplot as plt
from typing import List
import logging
from typing import Type, Union
def commit_all_pending_changes_to_version_control_system(self, base_path: str) -> None:
    """
        Commit all pending changes within the specified base path to the version control system.

        Args:
            base_path (str): The base path of the project directory.

        Returns:
            None

        Raises:
            Exception: If an error occurs during the commit process.

        This method performs the following steps:
        1. Navigate to the specified base path.
        2. Stage all modified files for commit using the version control system's command-line interface.
        3. Commit the staged changes with a descriptive commit message.
        4. Log the details of the commit process, including the committed files and the commit message.
        5. Handle any errors that may occur during the commit process and log them appropriately.
        """
    try:
        self.logger.debug(f'Navigating to base path: {base_path}')
        os.chdir(base_path)
        self.logger.debug('Staging all modified files for commit.')
        self._stage_all_modified_files()
        commit_message = 'Committing all pending changes.'
        self.logger.debug(f'Committing changes with message: {commit_message}')
        self._commit_changes(commit_message)
        self.logger.info('All pending changes committed successfully.')
    except subprocess.CalledProcessError as e:
        self.logger.error(f'Error occurred during the commit process: {str(e)}')
        raise
    except Exception as e:
        self.logger.error(f'Error occurred during the commit process: {str(e)}')
        raise