import time as time_mod
from http.cookies import SimpleCookie
from urllib.parse import quote as url_quote
from urllib.parse import unquote as url_unquote
from paste import request
def set_user_cookie(self, environ, userid, tokens, user_data):
    if not isinstance(tokens, str):
        tokens = ','.join(tokens)
    if self.include_ip:
        remote_addr = environ['REMOTE_ADDR']
    else:
        remote_addr = '0.0.0.0'
    ticket = AuthTicket(self.secret, userid, remote_addr, tokens=tokens, user_data=user_data, cookie_name=self.cookie_name, secure=self.secure)
    cur_domain = environ.get('HTTP_HOST', environ.get('SERVER_NAME'))
    wild_domain = '.' + cur_domain
    cookie_options = ''
    if self.secure:
        cookie_options += '; secure'
    if self.httponly:
        cookie_options += '; HttpOnly'
    cookies = []
    if self.no_domain_cookie:
        cookies.append(('Set-Cookie', '%s=%s; Path=/%s' % (self.cookie_name, ticket.cookie_value(), cookie_options)))
    if self.current_domain_cookie:
        cookies.append(('Set-Cookie', '%s=%s; Path=/; Domain=%s%s' % (self.cookie_name, ticket.cookie_value(), cur_domain, cookie_options)))
    if self.wildcard_cookie:
        cookies.append(('Set-Cookie', '%s=%s; Path=/; Domain=%s%s' % (self.cookie_name, ticket.cookie_value(), wild_domain, cookie_options)))
    return cookies