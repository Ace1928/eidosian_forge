from typing import Any, Dict, List
import pytest
import panel as pn
from panel.widgets import TextToSpeech, Utterance, Voice
def manualtest_get_app():
    text_to_speech = TextToSpeech(name='Speaker', value=TEXT, auto_speak=False)
    speaker_settings = pn.Param(text_to_speech, parameters=['value', 'speak', 'paused', 'speaking', 'pending', 'pause', 'resume', 'cancel', 'lang', 'voice', 'pitch', 'rate', 'volume', 'speak', 'value'], widgets={'speak': {'button_type': 'success'}, 'value': {'widget_type': pn.widgets.TextAreaInput, 'height': 300}}, expand_button=False, show_name=False)
    component = pn.Column(text_to_speech, speaker_settings, width=500, sizing_mode='fixed')
    template = pn.template.MaterialTemplate(title='Panel - TextToSpeech Widget')
    template.main.append(component)
    return template