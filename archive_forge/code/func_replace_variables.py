import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import types
def replace_variables(self, text, context):
    """
        Replace variable placeholders in script text with context-specific
        variables.

        Return the text passed in , but with variables replaced.

        :param text: The text in which to replace placeholder variables.
        :param context: The information for the environment creation request
                        being processed.
        """
    text = text.replace('__VENV_DIR__', context.env_dir)
    text = text.replace('__VENV_NAME__', context.env_name)
    text = text.replace('__VENV_PROMPT__', context.prompt)
    text = text.replace('__VENV_BIN_NAME__', context.bin_name)
    text = text.replace('__VENV_PYTHON__', context.env_exe)
    return text