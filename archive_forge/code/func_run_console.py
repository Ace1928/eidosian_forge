from base64 import urlsafe_b64encode
import hashlib
import json
import logging
import warnings
from string import ascii_letters, digits
import webbrowser
import wsgiref.simple_server
import wsgiref.util
import google.auth.transport.requests
import google.oauth2.credentials
import google_auth_oauthlib.helpers
def run_console(self, authorization_prompt_message=_DEFAULT_AUTH_PROMPT_MESSAGE, authorization_code_message=_DEFAULT_AUTH_CODE_MESSAGE, **kwargs):
    """Run the flow using the console strategy.

        .. deprecated:: 0.5.0
          Use :meth:`run_local_server` instead.

          The OAuth out-of-band (OOB) flow is deprecated. New clients will be unable to
          use this flow starting on Feb 28, 2022. This flow will be deprecated
          for all clients on Oct 3, 2022. Migrate to an alternative flow.

          See https://developers.googleblog.com/2022/02/making-oauth-flows-safer.html?m=1#disallowed-oob"

        The console strategy instructs the user to open the authorization URL
        in their browser. Once the authorization is complete the authorization
        server will give the user a code. The user then must copy & paste this
        code into the application. The code is then exchanged for a token.

        Args:
            authorization_prompt_message (str): The message to display to tell
                the user to navigate to the authorization URL.
            authorization_code_message (str): The message to display when
                prompting the user for the authorization code.
            kwargs: Additional keyword arguments passed through to
                :meth:`authorization_url`.

        Returns:
            google.oauth2.credentials.Credentials: The OAuth 2.0 credentials
                for the user.
        """
    kwargs.setdefault('prompt', 'consent')
    warnings.warn('New clients will be unable to use `InstalledAppFlow.run_console` starting on Feb 28, 2022. All clients will be unable to use this method starting on Oct 3, 2022. Use `InstalledAppFlow.run_local_server` instead. For details on the OOB flow deprecation, see https://developers.googleblog.com/2022/02/making-oauth-flows-safer.html?m=1#disallowed-oob', DeprecationWarning)
    self.redirect_uri = self._OOB_REDIRECT_URI
    auth_url, _ = self.authorization_url(**kwargs)
    print(authorization_prompt_message.format(url=auth_url))
    code = input(authorization_code_message)
    self.fetch_token(code=code)
    return self.credentials