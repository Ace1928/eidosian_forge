from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
import traceback
import datetime
def build_rfc_user_params(username, firstname, lastname, email, raw_password, useralias, user_type, raw_company, user_change, force):
    """Creates RFC parameters for Creating users"""
    params = dict()
    address = dict()
    password = dict()
    alias = dict()
    logondata = dict()
    company = dict()
    addressx = dict()
    passwordx = dict()
    logondatax = dict()
    companyx = dict()
    add_to_dict(params, 'USERNAME', username)
    add_to_dict(address, 'FIRSTNAME', firstname)
    add_to_dict(address, 'LASTNAME', lastname)
    add_to_dict(address, 'E_MAIL', email)
    add_to_dict(password, 'BAPIPWD', raw_password)
    add_to_dict(alias, 'USERALIAS', useralias)
    add_to_dict(logondata, 'GLTGV', datetime.date.today())
    add_to_dict(logondata, 'GLTGB', '20991231')
    add_to_dict(logondata, 'USTYP', user_type)
    add_to_dict(company, 'COMPANY', raw_company)
    params['LOGONDATA'] = logondata
    params['ADDRESS'] = address
    params['COMPANY'] = company
    params['ALIAS'] = alias
    params['PASSWORD'] = password
    if user_change and force:
        add_to_dict(addressx, 'FIRSTNAME', 'X')
        add_to_dict(addressx, 'LASTNAME', 'X')
        add_to_dict(addressx, 'E_MAIL', 'X')
        add_to_dict(passwordx, 'BAPIPWD', 'X')
        add_to_dict(logondatax, 'USTYP', 'X')
        add_to_dict(companyx, 'COMPANY', 'X')
        params['LOGONDATAX'] = logondatax
        params['ADDRESSX'] = addressx
        params['COMPANYX'] = companyx
        params['PASSWORDX'] = passwordx
    return params