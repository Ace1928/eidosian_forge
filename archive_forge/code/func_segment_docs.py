from wasabi import Printer
from ...errors import Errors
from ...tokens import Doc, Span
from ...training import iob_to_biluo
from ...util import get_lang_class, load_model
from .. import tags_to_entities
def segment_docs(input_data, n_sents, doc_delimiter):
    sent_delimiter = '\n\n'
    sents = input_data.split(sent_delimiter)
    docs = [sents[i:i + n_sents] for i in range(0, len(sents), n_sents)]
    input_data = ''
    for doc in docs:
        input_data += sent_delimiter + doc_delimiter
        input_data += sent_delimiter.join(doc)
    return input_data