import inspect
import os
import subprocess
import sys
import tempfile
from nltk.data import ZipFilePathPointer
from nltk.internals import find_dir, find_file, find_jars_within_path
from nltk.parse.api import ParserI
from nltk.parse.dependencygraph import DependencyGraph
from nltk.parse.util import taggedsents_to_conll
def generate_malt_command(self, inputfilename, outputfilename=None, mode=None):
    """
        This function generates the maltparser command use at the terminal.

        :param inputfilename: path to the input file
        :type inputfilename: str
        :param outputfilename: path to the output file
        :type outputfilename: str
        """
    cmd = ['java']
    cmd += self.additional_java_args
    classpaths_separator = ';' if sys.platform.startswith('win') else ':'
    cmd += ['-cp', classpaths_separator.join(self.malt_jars)]
    cmd += ['org.maltparser.Malt']
    if os.path.exists(self.model):
        cmd += ['-c', os.path.split(self.model)[-1]]
    else:
        cmd += ['-c', self.model]
    cmd += ['-i', inputfilename]
    if mode == 'parse':
        cmd += ['-o', outputfilename]
    cmd += ['-m', mode]
    return cmd