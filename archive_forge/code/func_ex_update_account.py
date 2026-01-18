import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_update_account(self, city: str=None, company_name: str=None, country: str=None, email: str=None, first_name: str=None, last_name: str=None, zip_code: str=None, job_title: str=None, mobile_number: str=None, phone_number: str=None, state_province: str=None, vat_number: str=None, dry_run: bool=False):
    """
        Updates the account information for the account that sends the request.

        :param      city: The city of the account owner.
        :type       city: ``str``

        :param      company_name: The name of the company for the account.
        permissions to perform the action.
        :type       company_name: ``str``

        :param      country: The country of the account owner.
        :type       country: ``str``

        :param      email: The email address for the account.
        :type       email: ``str``

        :param      first_name: The first name of the account owner.
        :type       first_name: ``str``

        :param      last_name: The last name of the account owner.
        :type       last_name: ``str``

        :param      zip_code: The ZIP code of the city.
        :type       zip_code: ``str``

        :param      job_title: The job title of the account owner.
        :type       job_title: ``str``

        :param      mobile_number: The mobile phone number of the account
        owner.
        :type       mobile_number: ``str``

        :param      phone_number: The landline phone number of the account
        owner.
        :type       phone_number: ``str``

        :param      state_province: The state/province of the account.
        :type       state_province: ``str``

        :param      vat_number: The value added tax (VAT) number for
        the account.
        :type       vat_number: ``str``

        :param      dry_run: the password of the account
        :type       dry_run: ``bool``

        :return: The new account information
        :rtype: ``dict``
        """
    action = 'UpdateAccount'
    data = {'DryRun': dry_run}
    if city is not None:
        data.update({'City': city})
    if company_name is not None:
        data.update({'CompanyName': company_name})
    if country is not None:
        data.update({'Country': country})
    if email is not None:
        data.update({'Email': email})
    if first_name is not None:
        data.update({'FirstName': first_name})
    if last_name is not None:
        data.update({'LastName': last_name})
    if zip_code is not None:
        data.update({'ZipCode': zip_code})
    if job_title is not None:
        data.update({'JobTitle': job_title})
    if mobile_number is not None:
        data.update({'MobileNumber': mobile_number})
    if phone_number is not None:
        data.update({'PhoneNumber': phone_number})
    if state_province is not None:
        data.update({'StateProvince': state_province})
    if vat_number is not None:
        data.update({'VatNumber': vat_number})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['Account']
    return response.json()