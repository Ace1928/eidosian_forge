import getpass
import io
import urllib.parse, urllib.request
from warnings import warn
from distutils.core import PyPIRCCommand
from distutils.errors import *
from distutils import log
def send_metadata(self):
    """ Send the metadata to the package index server.

            Well, do the following:
            1. figure who the user is, and then
            2. send the data as a Basic auth'ed POST.

            First we try to read the username/password from $HOME/.pypirc,
            which is a ConfigParser-formatted file with a section
            [distutils] containing username and password entries (both
            in clear text). Eg:

                [distutils]
                index-servers =
                    pypi

                [pypi]
                username: fred
                password: sekrit

            Otherwise, to figure who the user is, we offer the user three
            choices:

             1. use existing login,
             2. register as a new user, or
             3. set the password to a random string and email the user.

        """
    if self.has_config:
        choice = '1'
        username = self.username
        password = self.password
    else:
        choice = 'x'
        username = password = ''
    choices = '1 2 3 4'.split()
    while choice not in choices:
        self.announce('We need to know who you are, so please choose either:\n 1. use your existing login,\n 2. register as a new user,\n 3. have the server generate a new password for you (and email it to you), or\n 4. quit\nYour selection [default 1]: ', log.INFO)
        choice = input()
        if not choice:
            choice = '1'
        elif choice not in choices:
            print('Please choose one of the four options!')
    if choice == '1':
        while not username:
            username = input('Username: ')
        while not password:
            password = getpass.getpass('Password: ')
        auth = urllib.request.HTTPPasswordMgr()
        host = urllib.parse.urlparse(self.repository)[1]
        auth.add_password(self.realm, host, username, password)
        code, result = self.post_to_server(self.build_post_data('submit'), auth)
        self.announce('Server response (%s): %s' % (code, result), log.INFO)
        if code == 200:
            if self.has_config:
                self.distribution.password = password
            else:
                self.announce('I can store your PyPI login so future submissions will be faster.', log.INFO)
                self.announce('(the login will be stored in %s)' % self._get_rc_file(), log.INFO)
                choice = 'X'
                while choice.lower() not in 'yn':
                    choice = input('Save your login (y/N)?')
                    if not choice:
                        choice = 'n'
                if choice.lower() == 'y':
                    self._store_pypirc(username, password)
    elif choice == '2':
        data = {':action': 'user'}
        data['name'] = data['password'] = data['email'] = ''
        data['confirm'] = None
        while not data['name']:
            data['name'] = input('Username: ')
        while data['password'] != data['confirm']:
            while not data['password']:
                data['password'] = getpass.getpass('Password: ')
            while not data['confirm']:
                data['confirm'] = getpass.getpass(' Confirm: ')
            if data['password'] != data['confirm']:
                data['password'] = ''
                data['confirm'] = None
                print("Password and confirm don't match!")
        while not data['email']:
            data['email'] = input('   EMail: ')
        code, result = self.post_to_server(data)
        if code != 200:
            log.info('Server response (%s): %s', code, result)
        else:
            log.info('You will receive an email shortly.')
            log.info('Follow the instructions in it to complete registration.')
    elif choice == '3':
        data = {':action': 'password_reset'}
        data['email'] = ''
        while not data['email']:
            data['email'] = input('Your email address: ')
        code, result = self.post_to_server(data)
        log.info('Server response (%s): %s', code, result)