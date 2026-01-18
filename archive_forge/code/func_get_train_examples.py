import os
from ...utils import logging
from .utils import DataProcessor, InputExample
def get_train_examples(self, data_dir):
    """See base class."""
    lg = self.language if self.train_language is None else self.train_language
    lines = self._read_tsv(os.path.join(data_dir, f'XNLI-MT-1.0/multinli/multinli.train.{lg}.tsv'))
    examples = []
    for i, line in enumerate(lines):
        if i == 0:
            continue
        guid = f'train-{i}'
        text_a = line[0]
        text_b = line[1]
        label = 'contradiction' if line[2] == 'contradictory' else line[2]
        if not isinstance(text_a, str):
            raise ValueError(f'Training input {text_a} is not a string')
        if not isinstance(text_b, str):
            raise ValueError(f'Training input {text_b} is not a string')
        if not isinstance(label, str):
            raise ValueError(f'Training label {label} is not a string')
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples