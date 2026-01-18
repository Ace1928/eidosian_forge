import pytest
import panel as pn
from panel.widgets import Grammar, GrammarList, SpeechToText
from panel.widgets.speech_to_text import (
def manualtest_get_color_app():
    speech_to_text_color = SpeechToText(button_type='light', continuous=True)
    colors = ['aqua', 'azure', 'beige', 'bisque', 'black', 'blue', 'brown', 'chocolate', 'coral', 'crimson', 'cyan', 'fuchsia', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'indigo', 'ivory', 'khaki', 'lavender', 'lime', 'linen', 'magenta', 'maroon', 'moccasin', 'navy', 'olive', 'orange', 'orchid', 'peru', 'pink', 'plum', 'purple', 'red', 'salmon', 'sienna', 'silver', 'snow', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'white', 'yellow']
    src = '#JSGF V1.0; grammar colors; public <color> = ' + ' | '.join(colors) + ' ;'
    grammar_list = GrammarList()
    grammar_list.add_from_string(src, 1)
    speech_to_text_color.grammars = grammar_list
    colors_html = ', '.join([f"<span style='background:{color};'>{color}</span>" for color in colors])
    content_html = f'\n    <h1>Speech Color Changer</h1>\n\n    <p>Tap/click the microphone icon and say a color to change the background color of the app. Try {colors_html}\n    '
    content_panel = pn.pane.HTML(content_html, sizing_mode='stretch_width')
    app = pn.Column(sizing_mode='stretch_width', height=500, css_classes=['color-app'])
    style_panel = pn.pane.HTML(width=0, height=0, sizing_mode='fixed')
    result_panel = pn.pane.Markdown(sizing_mode='stretch_width')

    @pn.depends(speech_to_text_color, watch=True)
    def update_result_panel(results_last):
        results_last = results_last.lower()
        if results_last in colors:
            app.styles = dict(background=results_last)
            result_panel.object = 'Result received: ' + results_last
        else:
            app.styles = dict(background='white')
            result_panel.object = 'Result received: ' + results_last + ' (Not recognized)'
    app[:] = [style_panel, content_panel, speech_to_text_color, result_panel]
    return app