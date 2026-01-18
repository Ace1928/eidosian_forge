from _paramtreecfg import cfg
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.Parameter import PARAM_TYPES
from pyqtgraph.parametertree.parameterTypes import GroupParameter
def makeChild(chType, cfgDict):
    _encounteredTypes.add(chType)
    param = Parameter.create(name='widget', type=chType)
    if param.hasValue():
        param.setDefault(param.value())

    def setOpt(_param, _val):
        if isinstance(_val, str) and _val == '':
            _val = None
        param.setOpts(**{_param.name(): _val})
    optsChildren = []
    metaChildren = []
    for optName, optVals in cfgDict.items():
        child = Parameter.create(name=optName, **optVals)
        if ' ' in optName:
            metaChildren.append(child)
        else:
            optsChildren.append(child)
            child.sigValueChanged.connect(setOpt)
    for p in optsChildren:
        setOpt(p, p.value() if p.hasValue() else None)
    grp = Parameter.create(name=f'Sample {chType.title()}', type='group', children=metaChildren + [param] + optsChildren)
    grp.setOpts(expanded=False)
    return grp