from typing import Literal, Optional, Union, cast, Tuple
from .deprecation import AltairDeprecationWarning
from .html import spec_to_html
from ._importers import import_vl_convert, vl_version_for_vl_convert
import struct
import warnings
def spec_to_mimebundle(spec: dict, format: Literal['html', 'json', 'png', 'svg', 'pdf', 'vega', 'vega-lite'], mode: Optional[Literal['vega-lite']]=None, vega_version: Optional[str]=None, vegaembed_version: Optional[str]=None, vegalite_version: Optional[str]=None, embed_options: Optional[dict]=None, engine: Optional[Literal['vl-convert', 'altair_saver']]=None, **kwargs) -> Union[dict, Tuple[dict, dict]]:
    """Convert a vega-lite specification to a mimebundle

    The mimebundle type is controlled by the ``format`` argument, which can be
    one of the following ['html', 'json', 'png', 'svg', 'pdf', 'vega', 'vega-lite']

    Parameters
    ----------
    spec : dict
        a dictionary representing a vega-lite plot spec
    format : string {'html', 'json', 'png', 'svg', 'pdf', 'vega', 'vega-lite'}
        the file format to be saved.
    mode : string {'vega-lite'}
        The rendering mode.
    vega_version : string
        The version of vega.js to use
    vegaembed_version : string
        The version of vegaembed.js to use
    vegalite_version : string
        The version of vegalite.js to use. Only required if mode=='vega-lite'
    embed_options : dict (optional)
        The vegaEmbed options dictionary. Defaults to the embed options set with
        alt.renderers.set_embed_options().
        (See https://github.com/vega/vega-embed for details)
    engine: string {'vl-convert', 'altair_saver'}
        the conversion engine to use for 'png', 'svg', 'pdf', and 'vega' formats
    **kwargs :
        Additional arguments will be passed to the generating function

    Returns
    -------
    output : dict
        a mime-bundle representing the image

    Note
    ----
    The png, svg, pdf, and vega outputs require the altair_saver package
    """
    from altair.utils.display import compile_with_vegafusion, using_vegafusion
    from altair import renderers
    if mode != 'vega-lite':
        raise ValueError("mode must be 'vega-lite'")
    internal_mode: Literal['vega-lite', 'vega'] = mode
    if using_vegafusion():
        spec = compile_with_vegafusion(spec)
        internal_mode = 'vega'
    if embed_options is None:
        final_embed_options = renderers.options.get('embed_options', {})
    else:
        final_embed_options = embed_options
    embed_options = preprocess_embed_options(final_embed_options)
    if format in ['png', 'svg', 'pdf', 'vega']:
        format = cast(Literal['png', 'svg', 'pdf', 'vega'], format)
        return _spec_to_mimebundle_with_engine(spec, format, internal_mode, engine=engine, format_locale=embed_options.get('formatLocale', None), time_format_locale=embed_options.get('timeFormatLocale', None), **kwargs)
    if format == 'html':
        html = spec_to_html(spec, mode=internal_mode, vega_version=vega_version, vegaembed_version=vegaembed_version, vegalite_version=vegalite_version, embed_options=embed_options, **kwargs)
        return {'text/html': html}
    if format == 'vega-lite':
        if vegalite_version is None:
            raise ValueError('Must specify vegalite_version')
        return {'application/vnd.vegalite.v{}+json'.format(vegalite_version[0]): spec}
    if format == 'json':
        return {'application/json': spec}
    raise ValueError("format must be one of ['html', 'json', 'png', 'svg', 'pdf', 'vega', 'vega-lite']")