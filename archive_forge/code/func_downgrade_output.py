from __future__ import annotations
import json
import re
from traitlets.log import get_logger
from nbformat import v3, validator
from nbformat.corpus.words import generate_corpus_id as random_cell_id
from nbformat.notebooknode import NotebookNode
from .nbbase import nbformat, nbformat_minor
def downgrade_output(output):
    """downgrade a single code cell output to v3 from v4

    - pyout <- execute_result
    - pyerr <- error
    - output.data.mime/type -> output.type
    - un-mime-type keys
    - stream.stream <- stream.name
    """
    if output['output_type'] in {'execute_result', 'display_data'}:
        if output['output_type'] == 'execute_result':
            output['output_type'] = 'pyout'
            output['prompt_number'] = output.pop('execution_count', None)
        data = output.pop('data', {})
        if 'application/json' in data:
            data['application/json'] = json.dumps(data['application/json'])
        data = from_mime_key(data)
        output.update(data)
        from_mime_key(output.get('metadata', {}))
    elif output['output_type'] == 'error':
        output['output_type'] = 'pyerr'
    elif output['output_type'] == 'stream':
        output['stream'] = output.pop('name')
    return output