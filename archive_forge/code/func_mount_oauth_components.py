from __future__ import annotations
import json
import contextlib
from abc import ABC
from urllib.parse import urljoin
from lazyops.libs import lazyload
from fastapi import Request
from fastapi.background import BackgroundTasks
from ..utils.lazy import get_az_settings, get_az_mtg_api, get_az_resource_schema, logger
from ..utils.helpers import get_hashed_key, create_code_challenge, parse_scopes, encode_params_to_url
from typing import Optional, List, Dict, Any, Union, Type
def mount_oauth_components(self, login_path: Optional[str]='/login', logout_path: Optional[str]='/logout', server_identity: Optional[str]=None, server_identity_path: Optional[str]='/_identity', enable_authorize: Optional[bool]=True, authorize_path: Optional[str]='/authorize', enable_whoami: Optional[bool]=None, include_in_schema: Optional[bool]=None, user_class: Optional[Type['CurrentUser']]=None):
    """
        Mounts the OAuth Components
        """
    from fastapi import APIRouter, Depends, Query
    from fastapi.responses import RedirectResponse, JSONResponse, PlainTextResponse
    from .dependencies import get_current_user, CurrentUser as _CurrentUser
    router = APIRouter()

    @router.get(login_path, include_in_schema=include_in_schema)
    async def login(current_user: Optional[CurrentUser]=Depends(get_current_user(required=False, user_class=user_class)), redirect: Optional[str]=Query(None, description='The redirect page to use after login')):
        """
            Login Endpoint
            """
        if current_user is not None and current_user.is_valid:
            if redirect is not None:
                return RedirectResponse(self.get_app_redirection(redirect))
            return {'login': 'success', 'x-api-key': current_user.api_key}
        redirect_url = self.get_authorization_redirect_url(audience=self.settings.audience, scopes=self.settings.app_scopes)
        response = RedirectResponse(redirect_url)
        if redirect:
            response.set_cookie(key='x-auth-redirect', value=redirect, max_age=60, httponly=True)
        return response

    @router.get(self.callback_path, include_in_schema=include_in_schema)
    async def auth_callback(request: Request, code: Optional[str]=Query(None), background_tasks: BackgroundTasks=BackgroundTasks):
        """
            Callback Endpoint
            """
        current_user = await self.authorize_app_user(request=request, code=code, scopes=self.settings.app_scopes, background_tasks=background_tasks)
        if self.settings.is_development_env:
            logger.info(f'User {current_user.user_id} logged in')
        if (redir_value := request.cookies.get('x-auth-redirect')):
            redirect = self.get_app_redirection(redir_value)
            if self.settings.is_development_env:
                logger.info(f'Found redirect cookie: {redir_value} - Redirecting to {redirect}')
            response = RedirectResponse(redirect)
            response.delete_cookie('x-auth-redirect')
        else:
            response = JSONResponse({'login': 'success', 'x-api-key': current_user.api_key})
        response.set_cookie(**current_user.get_session_cookie_kwargs())
        return response

    @router.get(logout_path, include_in_schema=include_in_schema)
    async def logout(current_user: Optional[_CurrentUser]=Depends(get_current_user(required=False, user_class=user_class))):
        """
            Logout Endpoint
            """
        if current_user is None:
            return {'logout': 'no_user_found'}
        response = JSONResponse({'logout': 'success'})
        response.delete_cookie(**current_user.get_session_cookie_kwargs(is_delete=True))
        return response

    @router.get(self.callback_path, include_in_schema=include_in_schema)
    async def get_api_key(current_user: Optional[_CurrentUser]=Depends(get_current_user(required=False, user_class=user_class)), plaintext: Optional[bool]=Query(None, description='If True, will return the api key in plaintext')):
        """
            Get the API Key
            """
        if current_user is None or not current_user.is_valid:
            return 'null' if plaintext else {'api_key': 'no_user_found'}
        response = PlainTextResponse(content=current_user.api_key) if plaintext else JSONResponse(content={'api_key': current_user.api_key})
        response.set_cookie(**current_user.get_session_cookie_kwargs())
        return response
    if server_identity:

        @router.get(server_identity_path, include_in_schema=include_in_schema)
        async def get_server_identity(request: Request):
            """
                Get the Server Identity
                """
            return PlainTextResponse(content=server_identity)
    if enable_authorize:

        @router.get(authorize_path, include_in_schema=include_in_schema)
        async def authorize_user(current_user: _CurrentUser=Depends(get_current_user(user_class=user_class))):
            """
                Authorize the User or Client API by configuring the Cookies
                """
            if not current_user.is_valid:
                return {'authorize': 'invalid_user'}
            response = JSONResponse(content={'authorized': True, 'identity': server_identity or self.settings.app_name, 'environment': self.settings.app_env.value, 'api-key': current_user.api_key})
            response.set_cookie(**current_user.get_session_cookie_kwargs())
            return response
    if enable_whoami or self.settings.is_local_env:

        @router.get('/whoami')
        async def get_whoami_for_user(current_user: Optional[_CurrentUser]=Depends(get_current_user(required=False, user_class=user_class)), pretty: Optional[bool]=Query(None, description='If True, will return the user in a pretty format')):
            """
                Get the Whoami Data for the User
                """
            if current_user is None:
                return {'whoami': 'no_user_found'}
            data = current_user.get_whoami_data()
            if pretty:
                import yaml
                data = yaml.dump(data, default_flow_style=False, indent=2)
                return PlainTextResponse(content=data)
            return data
    self.app.include_router(router, tags=['oauth'])