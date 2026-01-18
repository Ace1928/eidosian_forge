from __future__ import annotations
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.utils import (
def analyze_text(text: str, nlp: Any=None, textstat: Any=None) -> dict:
    """Analyze text using textstat and spacy.

    Parameters:
        text (str): The text to analyze.
        nlp (spacy.lang): The spacy language model to use for visualization.

    Returns:
        (dict): A dictionary containing the complexity metrics and visualization
            files serialized to HTML string.
    """
    resp: Dict[str, Any] = {}
    if textstat is not None:
        text_complexity_metrics = {'flesch_reading_ease': textstat.flesch_reading_ease(text), 'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text), 'smog_index': textstat.smog_index(text), 'coleman_liau_index': textstat.coleman_liau_index(text), 'automated_readability_index': textstat.automated_readability_index(text), 'dale_chall_readability_score': textstat.dale_chall_readability_score(text), 'difficult_words': textstat.difficult_words(text), 'linsear_write_formula': textstat.linsear_write_formula(text), 'gunning_fog': textstat.gunning_fog(text), 'fernandez_huerta': textstat.fernandez_huerta(text), 'szigriszt_pazos': textstat.szigriszt_pazos(text), 'gutierrez_polini': textstat.gutierrez_polini(text), 'crawford': textstat.crawford(text), 'gulpease_index': textstat.gulpease_index(text), 'osman': textstat.osman(text)}
        resp.update({'text_complexity_metrics': text_complexity_metrics})
        resp.update(text_complexity_metrics)
    if nlp is not None:
        spacy = import_spacy()
        doc = nlp(text)
        dep_out = spacy.displacy.render(doc, style='dep', jupyter=False, page=True)
        ent_out = spacy.displacy.render(doc, style='ent', jupyter=False, page=True)
        text_visualizations = {'dependency_tree': dep_out, 'entities': ent_out}
        resp.update(text_visualizations)
    return resp