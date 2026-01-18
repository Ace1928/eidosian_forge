import inspect
import time
from typing import Iterable
from gradio_client.documentation import document_fn
import gradio as gr
import gradio as gr
def render_variables(history, base_theme, *args):
    primary_hue, secondary_hue, neutral_hue = args[0:3]
    primary_hues = args[3:3 + len(palette_range)]
    secondary_hues = args[3 + len(palette_range):3 + 2 * len(palette_range)]
    neutral_hues = args[3 + 2 * len(palette_range):3 + 3 * len(palette_range)]
    text_size, spacing_size, radius_size = args[3 + 3 * len(palette_range):6 + 3 * len(palette_range)]
    text_sizes = args[6 + 3 * len(palette_range):6 + 3 * len(palette_range) + len(size_range)]
    spacing_sizes = args[6 + 3 * len(palette_range) + len(size_range):6 + 3 * len(palette_range) + 2 * len(size_range)]
    radius_sizes = args[6 + 3 * len(palette_range) + 2 * len(size_range):6 + 3 * len(palette_range) + 3 * len(size_range)]
    main_fonts = args[6 + 3 * len(palette_range) + 3 * len(size_range):6 + 3 * len(palette_range) + 3 * len(size_range) + 4]
    main_is_google = args[6 + 3 * len(palette_range) + 3 * len(size_range) + 4:6 + 3 * len(palette_range) + 3 * len(size_range) + 8]
    mono_fonts = args[6 + 3 * len(palette_range) + 3 * len(size_range) + 8:6 + 3 * len(palette_range) + 3 * len(size_range) + 12]
    mono_is_google = args[6 + 3 * len(palette_range) + 3 * len(size_range) + 12:6 + 3 * len(palette_range) + 3 * len(size_range) + 16]
    remaining_args = args[6 + 3 * len(palette_range) + 3 * len(size_range) + 16:]
    final_primary_color = gr.themes.Color(*primary_hues)
    final_secondary_color = gr.themes.Color(*secondary_hues)
    final_neutral_color = gr.themes.Color(*neutral_hues)
    final_text_size = gr.themes.Size(*text_sizes)
    final_spacing_size = gr.themes.Size(*spacing_sizes)
    final_radius_size = gr.themes.Size(*radius_sizes)
    final_main_fonts = []
    font_weights = set()
    for attr, val in zip(flat_variables, remaining_args):
        if 'weight' in attr:
            font_weights.add(val)
    font_weights = sorted(font_weights)
    for main_font, is_google in zip(main_fonts, main_is_google):
        if not main_font:
            continue
        if is_google:
            main_font = gr.themes.GoogleFont(main_font, weights=font_weights)
        final_main_fonts.append(main_font)
    final_mono_fonts = []
    for mono_font, is_google in zip(mono_fonts, mono_is_google):
        if not mono_font:
            continue
        if is_google:
            mono_font = gr.themes.GoogleFont(mono_font, weights=font_weights)
        final_mono_fonts.append(mono_font)
    theme = gr.themes.Base(primary_hue=final_primary_color, secondary_hue=final_secondary_color, neutral_hue=final_neutral_color, text_size=final_text_size, spacing_size=final_spacing_size, radius_size=final_radius_size, font=final_main_fonts, font_mono=final_mono_fonts)
    theme.set(**dict(zip(flat_variables, remaining_args)))
    new_step = (base_theme, args)
    if len(history) == 0 or str(history[-1]) != str(new_step):
        history.append(new_step)
    return (history, theme._get_theme_css(), theme._stylesheets, generate_theme_code(base_theme, theme, (primary_hue, secondary_hue, neutral_hue, text_size, spacing_size, radius_size), list(zip(main_fonts, main_is_google)), list(zip(mono_fonts, mono_is_google))), theme)