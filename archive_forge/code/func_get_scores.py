import nltk
import os
import re
import itertools
import collections
import pkg_resources
def get_scores(self, hypothesis, references):
    """
        Compute precision, recall and f1 score between hypothesis and references

        Args:
          hypothesis: hypothesis summary, string
          references: reference summary/ies, either string or list of strings (if multiple)

        Returns:
          Return precision, recall and f1 score between hypothesis and references

        Raises:
          ValueError: raises exception if a type of hypothesis is different than the one of reference
          ValueError: raises exception if a len of hypothesis is different than the one of reference
        """
    if isinstance(hypothesis, str):
        hypothesis, references = ([hypothesis], [references])
    if type(hypothesis) != type(references):
        raise ValueError("'hyps' and 'refs' are not of the same type")
    if len(hypothesis) != len(references):
        raise ValueError("'hyps' and 'refs' do not have the same length")
    scores = {}
    has_rouge_n_metric = len([metric for metric in self.metrics if metric.split('-')[-1].isdigit()]) > 0
    if has_rouge_n_metric:
        scores = {**scores, **self._get_scores_rouge_n(hypothesis, references)}
    has_rouge_l_metric = len([metric for metric in self.metrics if metric.split('-')[-1].lower() == 'l']) > 0
    if has_rouge_l_metric:
        scores = {**scores, **self._get_scores_rouge_l_or_w(hypothesis, references, False)}
    has_rouge_w_metric = len([metric for metric in self.metrics if metric.split('-')[-1].lower() == 'w']) > 0
    if has_rouge_w_metric:
        scores = {**scores, **self._get_scores_rouge_l_or_w(hypothesis, references, True)}
    return scores