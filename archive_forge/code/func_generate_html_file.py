from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum
import os
import uuid
import webbrowser
import cirq_web
def generate_html_file(self, output_directory: str='./', file_name: str='bloch_sphere.html', open_in_browser: bool=False) -> str:
    """Generates a portable HTML file of the widget that
        can be run anywhere. Prints out the absolute path of the file to the console.

        Args:
            output_directory: the directory in which the output file will be
            generated. The default is the current directory ('./')

            file_name: the name of the output file. Default is 'bloch_sphere'

            open_in_browser: if True, opens the newly generated file automatically in the browser.

        Returns:
            The path of the HTML file in as a Path object.
        """
    client_code = self.get_client_code()
    contents = self._create_html_content(client_code)
    path_of_html_file = os.path.join(output_directory, file_name)
    with open(path_of_html_file, 'w', encoding='utf-8') as f:
        f.write(contents)
    if open_in_browser:
        webbrowser.open(path_of_html_file, new=2)
    return path_of_html_file