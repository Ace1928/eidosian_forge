import logging
from typing import List
from pathlib import Path
from . import extract_and_copy_jar, get_wheel_android_arch
from .. import Config, find_pyside_modules
def find_and_set_jars_dir(self):
    """Extract out and copy .jar files to {generated_files_path}
        """
    if not self.dry_run:
        logging.info(f'[DEPLOY] Extract and copy jar files from PySide6 wheel to {self.generated_files_path}')
        self.jars_dir = extract_and_copy_jar(wheel_path=self.wheel_pyside, generated_files_path=self.generated_files_path)