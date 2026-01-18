from __future__ import absolute_import, division, print_function
from os import environ
import platform
def get_blade(module):
    """Return System Object or Fail"""
    if HAS_DISTRO:
        user_agent = '%(base)s %(class)s/%(version)s (%(platform)s)' % {'base': USER_AGENT_BASE, 'class': __name__, 'version': VERSION, 'platform': distro.name(pretty=True)}
    else:
        user_agent = '%(base)s %(class)s/%(version)s (%(platform)s)' % {'base': USER_AGENT_BASE, 'class': __name__, 'version': VERSION, 'platform': platform.platform()}
    blade_name = module.params['fb_url']
    api = module.params['api_token']
    if HAS_PURITY_FB:
        if blade_name and api:
            blade = PurityFb(blade_name)
            blade.disable_verify_ssl()
            try:
                blade.login(api)
                versions = blade.api_version.list_versions().versions
                if API_AGENT_VERSION in versions:
                    blade._api_client.user_agent = user_agent
            except Exception:
                module.fail_json(msg='Pure Storage FlashBlade authentication failed. Check your credentials')
        elif environ.get('PUREFB_URL') and environ.get('PUREFB_API'):
            blade = PurityFb(environ.get('PUREFB_URL'))
            blade.disable_verify_ssl()
            try:
                blade.login(environ.get('PUREFB_API'))
                versions = blade.api_version.list_versions().versions
                if API_AGENT_VERSION in versions:
                    blade._api_client.user_agent = user_agent
            except Exception:
                module.fail_json(msg='Pure Storage FlashBlade authentication failed. Check your credentials')
        else:
            module.fail_json(msg='You must set PUREFB_URL and PUREFB_API environment variables or the fb_url and api_token module arguments')
    else:
        module.fail_json(msg='purity_fb SDK not installed.')
    return blade