import pytest
import panel as pn
from panel.widgets import Grammar, GrammarList, SpeechToText
from panel.widgets.speech_to_text import (
def manualtest_get_advanced_app():
    src = '#JSGF V1.0; grammar colors; public <color> = aqua | azure | beige | bisque | black | blue | brown | chocolate | coral | crimson | cyan | fuchsia | ghostwhite | gold | goldenrod | gray | green | indigo | ivory | khaki | lavender | lime | linen | magenta | maroon | moccasin | navy | olive | orange | orchid | peru | pink | plum | purple | red | salmon | sienna | silver | snow | tan | teal | thistle | tomato | turquoise | violet | white | yellow ;'
    speech_to_text = SpeechToText(button_type='success', continuous=True)
    grammar_list = GrammarList()
    grammar_list.add_from_string(src, 1)
    speech_to_text.grammars = grammar_list
    results_as_html_panel = pn.pane.Markdown(margin=(0, 15, 0, 15))

    @pn.depends(speech_to_text, watch=True)
    def update_results_html_panel(results):
        results_as_html_panel.object = speech_to_text.results_as_html
    speech_to_text_settings = pn.WidgetBox(pn.Param(speech_to_text, parameters=['start', 'stop', 'abort', 'grammars', 'lang', 'continuous', 'interim_results', 'max_alternatives', 'service_uri', 'started', 'results', 'value', 'started', 'audio_started', 'sound_started', 'speech_started', 'button_type', 'button_hide', 'button_started', 'button_not_started']))
    app = pn.Column(pn.pane.HTML("<h1>Speech to Text <img style='float:right;height:40px;width:164px;margin-right:40px' src='https://panel.holoviz.org/_static/logo_horizontal.png'></h1>", styles={'color': 'white', 'margin-left': '20px', 'background': 'black'}, margin=(0, 0, 15, 0)), speech_to_text, pn.Row(pn.Column(pn.pane.Markdown('## Settings'), speech_to_text_settings), pn.layout.VSpacer(width=25), pn.Column(pn.pane.Markdown('## Results'), results_as_html_panel)), width=800, sizing_mode='fixed')
    return app