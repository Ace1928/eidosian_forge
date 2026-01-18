import json
from typing import Optional, Dict
import jinja2
from altair.utils._importers import import_vl_convert, vl_version_for_vl_convert
def spec_to_html(spec: dict, mode: str, vega_version: Optional[str], vegaembed_version: Optional[str], vegalite_version: Optional[str]=None, base_url: str='https://cdn.jsdelivr.net/npm', output_div: str='vis', embed_options: Optional[dict]=None, json_kwds: Optional[dict]=None, fullhtml: bool=True, requirejs: bool=False, template: str='standard') -> str:
    """Embed a Vega/Vega-Lite spec into an HTML page

    Parameters
    ----------
    spec : dict
        a dictionary representing a vega-lite plot spec.
    mode : string {'vega' | 'vega-lite'}
        The rendering mode. This value is overridden by embed_options['mode'],
        if it is present.
    vega_version : string
        For html output, the version of vega.js to use.
    vegalite_version : string
        For html output, the version of vegalite.js to use.
    vegaembed_version : string
        For html output, the version of vegaembed.js to use.
    base_url : string (optional)
        The base url from which to load the javascript libraries.
    output_div : string (optional)
        The id of the div element where the plot will be shown.
    embed_options : dict (optional)
        Dictionary of options to pass to the vega-embed script. Default
        entry is {'mode': mode}.
    json_kwds : dict (optional)
        Dictionary of keywords to pass to json.dumps().
    fullhtml : boolean (optional)
        If True (default) then return a full html page. If False, then return
        an HTML snippet that can be embedded into an HTML page.
    requirejs : boolean (optional)
        If False (default) then load libraries from base_url using <script>
        tags. If True, then load libraries using requirejs
    template : jinja2.Template or string (optional)
        Specify the template to use (default = 'standard'). If template is a
        string, it must be one of {'universal', 'standard', 'inline'}. Otherwise, it
        can be a jinja2.Template object containing a custom template.

    Returns
    -------
    output : string
        an HTML string for rendering the chart.
    """
    embed_options = embed_options or {}
    json_kwds = json_kwds or {}
    mode = embed_options.setdefault('mode', mode)
    if mode not in ['vega', 'vega-lite']:
        raise ValueError("mode must be either 'vega' or 'vega-lite'")
    if vega_version is None:
        raise ValueError('must specify vega_version')
    if vegaembed_version is None:
        raise ValueError('must specify vegaembed_version')
    if mode == 'vega-lite' and vegalite_version is None:
        raise ValueError("must specify vega-lite version for mode='vega-lite'")
    render_kwargs = {}
    if template == 'inline':
        vlc = import_vl_convert()
        vl_version = vl_version_for_vl_convert()
        render_kwargs['vegaembed_script'] = vlc.javascript_bundle(vl_version=vl_version)
    jinja_template = TEMPLATES.get(template, template)
    if not hasattr(jinja_template, 'render'):
        raise ValueError('Invalid template: {0}'.format(jinja_template))
    return jinja_template.render(spec=json.dumps(spec, **json_kwds), embed_options=json.dumps(embed_options), mode=mode, vega_version=vega_version, vegalite_version=vegalite_version, vegaembed_version=vegaembed_version, base_url=base_url, output_div=output_div, fullhtml=fullhtml, requirejs=requirejs, **render_kwargs)