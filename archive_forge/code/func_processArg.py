import sys
from twisted.internet import reactor
from twisted.internet.protocol import Factory
from twisted.protocols import basic
def processArg(arg):
    if arg.lower() == 'no_ssl':
        global SSL_SUPPORT
        SSL_SUPPORT = False
        printMessage('NON-SSL')
    elif arg.lower() == 'no_uidl':
        global UIDL_SUPPORT
        UIDL_SUPPORT = False
        printMessage('NON-UIDL')
    elif arg.lower() == 'bad_resp':
        global INVALID_SERVER_RESPONSE
        INVALID_SERVER_RESPONSE = True
        printMessage('Invalid Server Response')
    elif arg.lower() == 'bad_cap_resp':
        global INVALID_CAPABILITY_RESPONSE
        INVALID_CAPABILITY_RESPONSE = True
        printMessage('Invalid Capability Response')
    elif arg.lower() == 'bad_login_resp':
        global INVALID_LOGIN_RESPONSE
        INVALID_LOGIN_RESPONSE = True
        printMessage('Invalid Capability Response')
    elif arg.lower() == 'deny':
        global DENY_CONNECTION
        DENY_CONNECTION = True
        printMessage('Deny Connection')
    elif arg.lower() == 'drop':
        global DROP_CONNECTION
        DROP_CONNECTION = True
        printMessage('Drop Connection')
    elif arg.lower() == 'bad_tls':
        global BAD_TLS_RESPONSE
        BAD_TLS_RESPONSE = True
        printMessage('Bad TLS Response')
    elif arg.lower() == 'timeout':
        global TIMEOUT_RESPONSE
        TIMEOUT_RESPONSE = True
        printMessage('Timeout Response')
    elif arg.lower() == 'to_deferred':
        global TIMEOUT_DEFERRED
        TIMEOUT_DEFERRED = True
        printMessage('Timeout Deferred Response')
    elif arg.lower() == 'slow':
        global SLOW_GREETING
        SLOW_GREETING = True
        printMessage('Slow Greeting')
    elif arg.lower() == '--help':
        print(usage)
        sys.exit()
    else:
        print(usage)
        sys.exit()