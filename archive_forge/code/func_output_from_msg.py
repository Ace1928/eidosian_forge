from __future__ import annotations
from nbformat.corpus.words import generate_corpus_id as random_cell_id
from nbformat.notebooknode import NotebookNode
def output_from_msg(msg):
    """Create a NotebookNode for an output from a kernel's IOPub message.

    Returns
    -------
    NotebookNode: the output as a notebook node.

    Raises
    ------
    ValueError: if the message is not an output message.

    """
    msg_type = msg['header']['msg_type']
    content = msg['content']
    if msg_type == 'execute_result':
        return new_output(output_type=msg_type, metadata=content['metadata'], data=content['data'], execution_count=content['execution_count'])
    if msg_type == 'stream':
        return new_output(output_type=msg_type, name=content['name'], text=content['text'])
    if msg_type == 'display_data':
        return new_output(output_type=msg_type, metadata=content['metadata'], data=content['data'])
    if msg_type == 'error':
        return new_output(output_type=msg_type, ename=content['ename'], evalue=content['evalue'], traceback=content['traceback'])
    raise ValueError('Unrecognized output msg type: %r' % msg_type)