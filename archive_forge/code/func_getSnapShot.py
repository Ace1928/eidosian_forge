def getSnapShot(self):
    """A snapshot of all the accounts and their status.

        @returns: A list of tuples, each of the form
            (string:accountName, boolean:isOnline,
            boolean:autoLogin, string:gatewayType)
        """
    data = []
    for account in self.accounts.values():
        data.append((account.accountName, account.isOnline(), account.autoLogin, account.gatewayType))
    return data