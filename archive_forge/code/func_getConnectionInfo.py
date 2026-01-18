def getConnectionInfo(self):
    connectioninfo = []
    for account in self.accounts.values():
        connectioninfo.append(account.isOnline())
    return connectioninfo