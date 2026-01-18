from statsmodels.compat.pandas import assert_series_equal
from io import StringIO
import warnings
import numpy as np
import numpy.testing as npt
import pandas as pd
import patsy
import pytest
from statsmodels.datasets import cpunish
from statsmodels.datasets.longley import load, load_pandas
from statsmodels.formula.api import ols
from statsmodels.formula.formulatools import make_hypotheses_matrices
from statsmodels.tools import add_constant
from statsmodels.tools.testing import assert_equal
def test_formula_labels():
    dta = StringIO('"type","income","education","prestige"\n"accountant","prof",62,86,82\n"pilot","prof",72,76,83\n"architect","prof",75,92,90\n"author","prof",55,90,76\n"chemist","prof",64,86,90\n"minister","prof",21,84,87\n"professor","prof",64,93,93\n"dentist","prof",80,100,90\n"reporter","wc",67,87,52\n"engineer","prof",72,86,88\n"undertaker","prof",42,74,57\n"lawyer","prof",76,98,89\n"physician","prof",76,97,97\n"welfare.worker","prof",41,84,59\n"teacher","prof",48,91,73\n"conductor","wc",76,34,38\n"contractor","prof",53,45,76\n"factory.owner","prof",60,56,81\n"store.manager","prof",42,44,45\n"banker","prof",78,82,92\n"bookkeeper","wc",29,72,39\n"mail.carrier","wc",48,55,34\n"insurance.agent","wc",55,71,41\n"store.clerk","wc",29,50,16\n"carpenter","bc",21,23,33\n"electrician","bc",47,39,53\n"RR.engineer","bc",81,28,67\n"machinist","bc",36,32,57\n"auto.repairman","bc",22,22,26\n"plumber","bc",44,25,29\n"gas.stn.attendant","bc",15,29,10\n"coal.miner","bc",7,7,15\n"streetcar.motorman","bc",42,26,19\n"taxi.driver","bc",9,19,10\n"truck.driver","bc",21,15,13\n"machine.operator","bc",21,20,24\n"barber","bc",16,26,20\n"bartender","bc",16,28,7\n"shoe.shiner","bc",9,17,3\n"cook","bc",14,22,16\n"soda.clerk","bc",12,30,6\n"watchman","bc",17,25,11\n"janitor","bc",7,20,8\n"policeman","bc",34,47,41\n"waiter","bc",8,32,10')
    from pandas import read_csv
    dta = read_csv(dta)
    model = ols('prestige ~ income + education', dta).fit()
    assert_equal(model.fittedvalues.index, dta.index)