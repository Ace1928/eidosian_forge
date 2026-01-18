import pytest
import panel as pn
from panel.widgets import Grammar, GrammarList, SpeechToText
from panel.widgets.speech_to_text import (
def test_add_from_string_to_speech_grammar_list():
    src = '#JSGF V1.0; grammar colors; public <color> = aqua | azure | beige | bisque | black | blue | brown | chocolate | coral | crimson | cyan | fuchsia | ghostwhite | gold | goldenrod | gray | green | indigo | ivory | khaki | lavender | lime | linen | magenta | maroon | moccasin | navy | olive | orange | orchid | peru | pink | plum | purple | red | salmon | sienna | silver | snow | tan | teal | thistle | tomato | turquoise | violet | white | yellow ;'
    weight = 0.5
    grammar_list = GrammarList()
    result = grammar_list.add_from_string(src, weight)
    serialized = grammar_list.serialize()
    assert isinstance(result, Grammar)
    assert result in grammar_list
    assert serialized == [{'src': src, 'weight': weight}]