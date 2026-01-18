import collections
import importlib
import os
import re
import sys
from fnmatch import fnmatch
from pathlib import Path
from os.path import isfile, join
from urllib.parse import parse_qs
import flask
from . import _validate
from ._utils import AttributeDict
from ._get_paths import get_relative_path
from ._callback_context import context_value
from ._get_app import get_app
def register_page(module, path=None, path_template=None, name=None, order=None, title=None, description=None, image=None, image_url=None, redirect_from=None, layout=None, **kwargs):
    """
    Assigns the variables to `dash.page_registry` as an `OrderedDict`
    (ordered by `order`).

    `dash.page_registry` is used by `pages_plugin` to set up the layouts as
    a multi-page Dash app. This includes the URL routing callbacks
    (using `dcc.Location`) and the HTML templates to include title,
    meta description, and the meta description image.

    `dash.page_registry` can also be used by Dash developers to create the
    page navigation links or by template authors.

    - `module`:
       The module path where this page's `layout` is defined. Often `__name__`.

    - `path`:
       URL Path, e.g. `/` or `/home-page`.
       If not supplied, will be inferred from the `path_template` or `module`,
       e.g. based on path_template: `/asset/<asset_id` to `/asset/none`
       e.g. based on module: `pages.weekly_analytics` to `/weekly-analytics`

    - `relative_path`:
        The path with `requests_pathname_prefix` prefixed before it.
        Use this path when specifying local URL paths that will work
        in environments regardless of what `requests_pathname_prefix` is.
        In some deployment environments, like Dash Enterprise,
        `requests_pathname_prefix` is set to the application name,
        e.g. `my-dash-app`.
        When working locally, `requests_pathname_prefix` might be unset and
        so a relative URL like `/page-2` can just be `/page-2`.
        However, when the app is deployed to a URL like `/my-dash-app`, then
        `relative_path` will be `/my-dash-app/page-2`.

    - `path_template`:
       Add variables to a URL by marking sections with <variable_name>. The layout function
       then receives the <variable_name> as a keyword argument.
       e.g. path_template= "/asset/<asset_id>"
            then if pathname in browser is "/assets/a100" then layout will receive **{"asset_id":"a100"}

    - `name`:
       The name of the link.
       If not supplied, will be inferred from `module`,
       e.g. `pages.weekly_analytics` to `Weekly analytics`

    - `order`:
       The order of the pages in `page_registry`.
       If not supplied, then the filename is used and the page with path `/` has
       order `0`

    - `title`:
       (string or function) The name of the page <title>. That is, what appears in the browser title.
       If not supplied, will use the supplied `name` or will be inferred by module,
       e.g. `pages.weekly_analytics` to `Weekly analytics`

    - `description`:
       (string or function) The <meta type="description"></meta>.
       If not supplied, then nothing is supplied.

    - `image`:
       The meta description image used by social media platforms.
       If not supplied, then it looks for the following images in `assets/`:
        - A page specific image: `assets/<module>.<extension>` is used, e.g. `assets/weekly_analytics.png`
        - A generic app image at `assets/app.<extension>`
        - A logo at `assets/logo.<extension>`
        When inferring the image file, it will look for the following extensions:
        APNG, AVIF, GIF, JPEG, JPG, PNG, SVG, WebP.

    -  `image_url`:
       Overrides the image property and sets the `<image>` meta tag to the provided image URL.

    - `redirect_from`:
       A list of paths that should redirect to this page.
       For example: `redirect_from=['/v2', '/v3']`

    - `layout`:
       The layout function or component for this page.
       If not supplied, then looks for `layout` from within the supplied `module`.

    - `**kwargs`:
       Arbitrary keyword arguments that can be stored

    ***

    `page_registry` stores the original property that was passed in under
    `supplied_<property>` and the coerced property under `<property>`.
    For example, if this was called:
    ```
    register_page(
        'pages.historical_outlook',
        name='Our historical view',
        custom_key='custom value'
    )
    ```
    Then this will appear in `page_registry`:
    ```
    OrderedDict([
        (
            'pages.historical_outlook',
            dict(
                module='pages.historical_outlook',

                supplied_path=None,
                path='/historical-outlook',

                supplied_name='Our historical view',
                name='Our historical view',

                supplied_title=None,
                title='Our historical view'

                supplied_layout=None,
                layout=<function pages.historical_outlook.layout>,

                custom_key='custom value'
            )
        ),
    ])
    ```
    """
    if context_value.get().get('ignore_register_page'):
        return
    _validate.validate_use_pages(CONFIG)
    page = dict(module=_validate.validate_module_name(module), supplied_path=path, path_template=path_template, path=path if path is not None else _infer_path(module, path_template), supplied_name=name, name=name if name is not None else _module_name_to_page_name(module))
    page.update(supplied_title=title, title=title if title is not None else page['name'])
    page.update(description=description if description else '', order=order, supplied_order=order, supplied_layout=layout, **kwargs)
    page.update(supplied_image=image, image=image if image is not None else _infer_image(module), image_url=image_url)
    page.update(redirect_from=_set_redirect(redirect_from, page['path']))
    PAGE_REGISTRY[module] = page
    if page['path_template']:
        _validate.validate_template(page['path_template'])
    if layout is not None:
        PAGE_REGISTRY[module]['layout'] = layout
    order_supplied = any((p['supplied_order'] is not None for p in PAGE_REGISTRY.values()))
    for p in PAGE_REGISTRY.values():
        p['order'] = 0 if p['path'] == '/' and (not order_supplied) else p['supplied_order']
        p['relative_path'] = get_relative_path(p['path'])
    for page in sorted(PAGE_REGISTRY.values(), key=lambda i: (i['order'] is None, i['order'] if isinstance(i['order'], (int, float)) else float('inf'), str(i['order']), i['module'])):
        PAGE_REGISTRY.move_to_end(page['module'])