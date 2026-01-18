import copy
import json
import os
from parlai.core.teachers import DialogTeacher
from .build import build
def convert_to_dialogData(premise_raw, hypo_raw, answer_raw, dialog_format=False, binary_classes=False):
    """
    Convert from NLI context to dialog text.

    :param premise_raw: raw premise extracted from jsonl file.
    :param hypo_raw: raw hypothesis extracted from jsonl file.
    :param answer_raw: raw answer extracted from jsonl file.
    :param dialog_format: if set True, omit the special tokens 'Hypothesis' and 'Premise' in the text.
    :param binary_classes: if set True, bucketize (neutral, entailment) into one (no_contradiction)
    :return: a tuple (question, answer, clas)
        - ``question`` (str) is a query and possibly context
        - ``answers`` (iter) is an iterable of label(s) for that query
        - ``clas`` (iter) is an iterable of label candidates that the student can choose from
    """
    premise_raw = premise_raw.strip('\n').strip('\t')
    hypo_raw = hypo_raw.strip('\n').strip('\t')
    clas = MULTINLI_LABELS
    if binary_classes:
        answer_raw = BICLASS_DICT[answer_raw]
        clas = BICLASS_LABELS
    if not dialog_format:
        premise_raw = MULTINLI_PREMISE_PREFIX + premise_raw
        hypo_raw = MULTINLI_HYPO_PREFIX + hypo_raw
    question = premise_raw + '\n' + hypo_raw
    answers = [answer_raw]
    return (question, answers, clas)