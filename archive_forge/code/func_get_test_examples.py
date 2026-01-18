import os
from ...utils import logging
from .utils import DataProcessor, InputExample
def get_test_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, 'XNLI-1.0/xnli.test.tsv'))
    examples = []
    for i, line in enumerate(lines):
        if i == 0:
            continue
        language = line[0]
        if language != self.language:
            continue
        guid = f'test-{i}'
        text_a = line[6]
        text_b = line[7]
        label = line[1]
        if not isinstance(text_a, str):
            raise ValueError(f'Training input {text_a} is not a string')
        if not isinstance(text_b, str):
            raise ValueError(f'Training input {text_b} is not a string')
        if not isinstance(label, str):
            raise ValueError(f'Training label {label} is not a string')
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples