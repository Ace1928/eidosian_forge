from __future__ import unicode_literals
from os import path
from pybtex import Engine
def format_from_files(self, bib_files_or_filenames, style, citations=['*'], bib_format=None, bib_encoding=None, output_encoding=None, bst_encoding=None, min_crossrefs=2, output_filename=None, add_output_suffix=False, **kwargs):
    """
        Read the bigliography data from the given files and produce a formated
        bibliography.

        :param bib_files_or_filenames: A list of file names or file objects.
        :param style: The name of the formatting style.
        :param citations: A list of citation keys.
        :param bib_format: The name of the bibliography format. The default
            format is ``bibtex``.
        :param bib_encoding: Encoding of bibliography files.
        :param output_encoding: Encoding that will be used by the output backend.
        :param bst_encoding: Encoding of the ``.bst`` file.
        :param min_crossrefs: Include cross-referenced entries after this many
            crossrefs. See BibTeX manual for details.
        :param output_filename: If ``None``, the result will be returned as a
            string. Else, the result will be written to the specified file.
        :param add_output_suffix: Append a ``.bbl`` suffix to the output file name.
        """
    from io import StringIO
    import pybtex.io
    from pybtex.bibtex import bst
    from pybtex.bibtex.interpreter import Interpreter
    if bib_format is None:
        from pybtex.database.input.bibtex import Parser as bib_format
    bst_filename = style + path.extsep + 'bst'
    bst_script = bst.parse_file(bst_filename, bst_encoding)
    interpreter = Interpreter(bib_format, bib_encoding)
    bbl_data = interpreter.run(bst_script, citations, bib_files_or_filenames, min_crossrefs=min_crossrefs)
    if add_output_suffix:
        output_filename = output_filename + '.bbl'
    if output_filename:
        output_file = pybtex.io.open_unicode(output_filename, 'w', encoding=output_encoding)
    else:
        output_file = StringIO()
    with output_file:
        output_file.write(bbl_data)
        if isinstance(output_file, StringIO):
            return output_file.getvalue()