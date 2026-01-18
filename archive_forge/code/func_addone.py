import pyomo.environ as pyo
def addone(self, scen):
    """Add a scenario to the set

        Args:
            scen (ParmestScen): the scenario to add
        """
    assert isinstance(self._scens, list)
    self._scens.append(scen)