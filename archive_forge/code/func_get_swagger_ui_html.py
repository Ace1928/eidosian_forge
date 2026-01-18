import json
from typing import Any, Dict, Optional
from fastapi.encoders import jsonable_encoder
from starlette.responses import HTMLResponse
from typing_extensions import Annotated, Doc  # type: ignore [attr-defined]
def get_swagger_ui_html(*, openapi_url: Annotated[str, Doc('\n            The OpenAPI URL that Swagger UI should load and use.\n\n            This is normally done automatically by FastAPI using the default URL\n            `/openapi.json`.\n            ')], title: Annotated[str, Doc('\n            The HTML `<title>` content, normally shown in the browser tab.\n            ')], swagger_js_url: Annotated[str, Doc('\n            The URL to use to load the Swagger UI JavaScript.\n\n            It is normally set to a CDN URL.\n            ')]='https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js', swagger_css_url: Annotated[str, Doc('\n            The URL to use to load the Swagger UI CSS.\n\n            It is normally set to a CDN URL.\n            ')]='https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css', swagger_favicon_url: Annotated[str, Doc('\n            The URL of the favicon to use. It is normally shown in the browser tab.\n            ')]='https://fastapi.tiangolo.com/img/favicon.png', oauth2_redirect_url: Annotated[Optional[str], Doc('\n            The OAuth2 redirect URL, it is normally automatically handled by FastAPI.\n            ')]=None, init_oauth: Annotated[Optional[Dict[str, Any]], Doc('\n            A dictionary with Swagger UI OAuth2 initialization configurations.\n            ')]=None, swagger_ui_parameters: Annotated[Optional[Dict[str, Any]], Doc('\n            Configuration parameters for Swagger UI.\n\n            It defaults to [swagger_ui_default_parameters][fastapi.openapi.docs.swagger_ui_default_parameters].\n            ')]=None) -> HTMLResponse:
    """
    Generate and return the HTML  that loads Swagger UI for the interactive
    API docs (normally served at `/docs`).

    You would only call this function yourself if you needed to override some parts,
    for example the URLs to use to load Swagger UI's JavaScript and CSS.

    Read more about it in the
    [FastAPI docs for Configure Swagger UI](https://fastapi.tiangolo.com/how-to/configure-swagger-ui/)
    and the [FastAPI docs for Custom Docs UI Static Assets (Self-Hosting)](https://fastapi.tiangolo.com/how-to/custom-docs-ui-assets/).
    """
    current_swagger_ui_parameters = swagger_ui_default_parameters.copy()
    if swagger_ui_parameters:
        current_swagger_ui_parameters.update(swagger_ui_parameters)
    html = f"""\n    <!DOCTYPE html>\n    <html>\n    <head>\n    <link type="text/css" rel="stylesheet" href="{swagger_css_url}">\n    <link rel="shortcut icon" href="{swagger_favicon_url}">\n    <title>{title}</title>\n    </head>\n    <body>\n    <div id="swagger-ui">\n    </div>\n    <script src="{swagger_js_url}"></script>\n    <!-- `SwaggerUIBundle` is now available on the page -->\n    <script>\n    const ui = SwaggerUIBundle({{\n        url: '{openapi_url}',\n    """
    for key, value in current_swagger_ui_parameters.items():
        html += f'{json.dumps(key)}: {json.dumps(jsonable_encoder(value))},\n'
    if oauth2_redirect_url:
        html += f"oauth2RedirectUrl: window.location.origin + '{oauth2_redirect_url}',"
    html += '\n    presets: [\n        SwaggerUIBundle.presets.apis,\n        SwaggerUIBundle.SwaggerUIStandalonePreset\n        ],\n    })'
    if init_oauth:
        html += f'\n        ui.initOAuth({json.dumps(jsonable_encoder(init_oauth))})\n        '
    html += '\n    </script>\n    </body>\n    </html>\n    '
    return HTMLResponse(html)