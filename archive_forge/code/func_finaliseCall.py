import ctypes, logging
from OpenGL import platform, error
from OpenGL._configflags import STORE_POINTERS, ERROR_ON_COPY, SIZE_1_ARRAY_UNPACK
from OpenGL import converters
from OpenGL.converters import DefaultCConverter
from OpenGL.converters import returnCArgument,returnPyArgument
from OpenGL.latebind import LateBind
from OpenGL.arrays import arrayhelpers, arraydatatype
from OpenGL._null import NULL
from OpenGL import acceleratesupport
def finaliseCall(self):
    """Produce specialised versions of call for finalised wrapper object

        This returns a version of __call__ that only does that work which is
        required by the particular wrapper object

        This is essentially a huge set of expanded nested functions, very
        inelegant...
        """
    pyConverters = getattr(self, 'pyConverters', None)
    cConverters = getattr(self, 'cConverters', None)
    cResolvers = getattr(self, 'cResolvers', None)
    wrappedOperation = self.wrappedOperation
    storeValues = getattr(self, 'storeValues', None)
    returnValues = getattr(self, 'returnValues', None)
    if pyConverters:
        if cWrapper:
            calculate_pyArgs = PyArgCalculator(self, pyConverters)
        else:
            pyConverters_mapped = [(i, converter, converter is None) for i, converter in enumerate(pyConverters)]
            pyConverters_length = len([p for p in pyConverters if not getattr(p, 'optional', False)])

            def calculate_pyArgs(args):
                if pyConverters_length > len(args):
                    raise ValueError('%s requires %r arguments (%s), received %s: %r' % (wrappedOperation.__name__, pyConverters_length, ', '.join(self.pyConverterNames), len(args), args))
                for index, converter, isNone in pyConverters_mapped:
                    if isNone:
                        yield args[index]
                    else:
                        try:
                            yield converter(args[index], self, args)
                        except IndexError as err:
                            yield NULL
                        except Exception as err:
                            if hasattr(err, 'args'):
                                err.args += (converter,)
                            raise
    else:
        calculate_pyArgs = None
    if cConverters:
        if cWrapper:
            calculate_cArgs = CArgCalculator(self, cConverters)
        else:
            cConverters_mapped = [(i, converter, hasattr(converter, '__call__')) for i, converter in enumerate(cConverters)]

            def calculate_cArgs(pyArgs):
                for index, converter, canCall in cConverters_mapped:
                    if canCall:
                        try:
                            yield converter(pyArgs, index, self)
                        except Exception as err:
                            if hasattr(err, 'args'):
                                err.args += ('Failure in cConverter %r' % converter, pyArgs, index, self)
                            raise
                    else:
                        yield converter
    else:
        calculate_cArgs = None
    if cResolvers:
        if cWrapper:
            calculate_cArguments = CArgumentCalculator(cResolvers)
        else:
            cResolvers_mapped = list(enumerate(cResolvers))

            def calculate_cArguments(cArgs):
                for i, converter in cResolvers_mapped:
                    if converter is None:
                        yield cArgs[i]
                    else:
                        try:
                            yield converter(cArgs[i])
                        except Exception as err:
                            err.args += (converter,)
                            raise
    else:
        calculate_cArguments = None
    if cWrapper:
        return cWrapper(wrappedOperation, calculate_pyArgs=calculate_pyArgs, calculate_cArgs=calculate_cArgs, calculate_cArguments=calculate_cArguments, storeValues=storeValues, returnValues=returnValues)
    if pyConverters:
        if cConverters:
            if cResolvers:
                if storeValues:
                    if returnValues:

                        def wrapperCall(*args):
                            """Wrapper with all possible operations"""
                            pyArgs = tuple(calculate_pyArgs(args))
                            cArgs = tuple(calculate_cArgs(pyArgs))
                            cArguments = tuple(calculate_cArguments(cArgs))
                            try:
                                result = wrappedOperation(*cArguments)
                            except ctypes.ArgumentError as err:
                                err.args = err.args + (cArguments,)
                                raise err
                            except error.GLError as err:
                                err.cArgs = cArgs
                                err.pyArgs = pyArgs
                                raise err
                            storeValues(result, self, pyArgs, cArgs)
                            return returnValues(result, self, pyArgs, cArgs)
                        return wrapperCall
                    else:

                        def wrapperCall(*args):
                            """Wrapper with all save returnValues"""
                            pyArgs = tuple(calculate_pyArgs(args))
                            cArgs = tuple(calculate_cArgs(pyArgs))
                            cArguments = tuple(calculate_cArguments(cArgs))
                            try:
                                result = wrappedOperation(*cArguments)
                            except ctypes.ArgumentError as err:
                                err.args = err.args + (cArguments,)
                                raise err
                            except error.GLError as err:
                                err.cArgs = cArgs
                                err.pyArgs = pyArgs
                                raise err
                            storeValues(result, self, pyArgs, cArgs)
                            return result
                        return wrapperCall
                elif returnValues:

                    def wrapperCall(*args):
                        """Wrapper with all save storeValues"""
                        pyArgs = tuple(calculate_pyArgs(args))
                        cArgs = tuple(calculate_cArgs(pyArgs))
                        cArguments = tuple(calculate_cArguments(cArgs))
                        try:
                            result = wrappedOperation(*cArguments)
                        except ctypes.ArgumentError as err:
                            err.args = err.args + (cArguments,)
                            raise err
                        except error.GLError as err:
                            err.cArgs = cArgs
                            err.pyArgs = pyArgs
                            raise err
                        return returnValues(result, self, pyArgs, cArgs)
                    return wrapperCall
                else:

                    def wrapperCall(*args):
                        """Wrapper with all save returnValues and storeValues"""
                        pyArgs = tuple(calculate_pyArgs(args))
                        cArgs = tuple(calculate_cArgs(pyArgs))
                        cArguments = tuple(calculate_cArguments(cArgs))
                        try:
                            result = wrappedOperation(*cArguments)
                        except ctypes.ArgumentError as err:
                            err.args = err.args + (cArguments,)
                            raise err
                        except error.GLError as err:
                            err.cArgs = cArgs
                            err.pyArgs = pyArgs
                            raise err
                        return result
                    return wrapperCall
            elif storeValues:
                if returnValues:

                    def wrapperCall(*args):
                        """Wrapper with all possible operations"""
                        pyArgs = tuple(calculate_pyArgs(args))
                        cArgs = tuple(calculate_cArgs(pyArgs))
                        cArguments = cArgs
                        try:
                            result = wrappedOperation(*cArguments)
                        except ctypes.ArgumentError as err:
                            err.args = err.args + (cArguments,)
                            raise err
                        except error.GLError as err:
                            err.cArgs = cArgs
                            err.pyArgs = pyArgs
                            raise err
                        storeValues(result, self, pyArgs, cArgs)
                        return returnValues(result, self, pyArgs, cArgs)
                    return wrapperCall
                else:

                    def wrapperCall(*args):
                        """Wrapper with all save returnValues"""
                        pyArgs = tuple(calculate_pyArgs(args))
                        cArgs = tuple(calculate_cArgs(pyArgs))
                        cArguments = cArgs
                        try:
                            result = wrappedOperation(*cArguments)
                        except ctypes.ArgumentError as err:
                            err.args = err.args + (cArguments,)
                            raise err
                        except error.GLError as err:
                            err.cArgs = cArgs
                            err.pyArgs = pyArgs
                            raise err
                        storeValues(result, self, pyArgs, cArgs)
                        return result
                    return wrapperCall
            elif returnValues:

                def wrapperCall(*args):
                    """Wrapper with all save storeValues"""
                    pyArgs = tuple(calculate_pyArgs(args))
                    cArgs = tuple(calculate_cArgs(pyArgs))
                    cArguments = cArgs
                    try:
                        result = wrappedOperation(*cArguments)
                    except ctypes.ArgumentError as err:
                        err.args = err.args + (cArguments,)
                        raise
                    except error.GLError as err:
                        err.cArgs = cArgs
                        err.pyArgs = pyArgs
                        raise err
                    return returnValues(result, self, pyArgs, cArgs)
                return wrapperCall
            else:

                def wrapperCall(*args):
                    """Wrapper with all save returnValues and storeValues"""
                    pyArgs = tuple(calculate_pyArgs(args))
                    cArgs = tuple(calculate_cArgs(pyArgs))
                    cArguments = cArgs
                    try:
                        result = wrappedOperation(*cArguments)
                    except ctypes.ArgumentError as err:
                        err.args = err.args + (cArguments,)
                        raise
                    except error.GLError as err:
                        err.cArgs = cArgs
                        err.pyArgs = pyArgs
                        raise err
                    return result
                return wrapperCall
        elif cResolvers:
            if storeValues:
                if returnValues:

                    def wrapperCall(*args):
                        """Wrapper with all possible operations"""
                        pyArgs = tuple(calculate_pyArgs(args))
                        cArgs = pyArgs
                        cArguments = tuple(calculate_cArguments(cArgs))
                        try:
                            result = wrappedOperation(*cArguments)
                        except ctypes.ArgumentError as err:
                            err.args = err.args + (cArguments,)
                            raise err
                        except error.GLError as err:
                            err.cArgs = cArgs
                            err.pyArgs = pyArgs
                            raise err
                        storeValues(result, self, pyArgs, cArgs)
                        return returnValues(result, self, pyArgs, cArgs)
                    return wrapperCall
                else:

                    def wrapperCall(*args):
                        """Wrapper with all save returnValues"""
                        pyArgs = tuple(calculate_pyArgs(args))
                        cArgs = pyArgs
                        cArguments = tuple(calculate_cArguments(cArgs))
                        try:
                            result = wrappedOperation(*cArguments)
                        except ctypes.ArgumentError as err:
                            err.args = err.args + (cArguments,)
                            raise err
                        except error.GLError as err:
                            err.cArgs = cArgs
                            err.pyArgs = pyArgs
                            raise err
                        storeValues(result, self, pyArgs, cArgs)
                        return result
                    return wrapperCall
            elif returnValues:

                def wrapperCall(*args):
                    """Wrapper with all save storeValues"""
                    pyArgs = tuple(calculate_pyArgs(args))
                    cArgs = pyArgs
                    cArguments = tuple(calculate_cArguments(cArgs))
                    try:
                        result = wrappedOperation(*cArguments)
                    except ctypes.ArgumentError as err:
                        err.args = err.args + (cArguments,)
                        raise err
                    except error.GLError as err:
                        err.cArgs = cArgs
                        err.pyArgs = pyArgs
                        raise err
                    return returnValues(result, self, pyArgs, cArgs)
                return wrapperCall
            else:

                def wrapperCall(*args):
                    """Wrapper with all save returnValues and storeValues"""
                    pyArgs = tuple(calculate_pyArgs(args))
                    cArgs = pyArgs
                    cArguments = tuple(calculate_cArguments(cArgs))
                    try:
                        result = wrappedOperation(*cArguments)
                    except ctypes.ArgumentError as err:
                        err.args = err.args + (cArguments,)
                        raise err
                    except error.GLError as err:
                        err.cArgs = cArgs
                        err.pyArgs = pyArgs
                        raise err
                    return result
                return wrapperCall
        elif storeValues:
            if returnValues:

                def wrapperCall(*args):
                    """Wrapper with all possible operations"""
                    pyArgs = tuple(calculate_pyArgs(args))
                    cArguments = pyArgs
                    try:
                        result = wrappedOperation(*cArguments)
                    except ctypes.ArgumentError as err:
                        err.args = err.args + (cArguments,)
                        raise err
                    except error.GLError as err:
                        err.cArgs = cArguments
                        err.pyArgs = pyArgs
                        raise err
                    storeValues(result, self, pyArgs, cArguments)
                    return returnValues(result, self, pyArgs, cArguments)
                return wrapperCall
            else:

                def wrapperCall(*args):
                    """Wrapper with all save returnValues"""
                    pyArgs = tuple(calculate_pyArgs(args))
                    cArguments = pyArgs
                    try:
                        result = wrappedOperation(*cArguments)
                    except ctypes.ArgumentError as err:
                        err.args = err.args + (cArguments,)
                        raise err
                    except error.GLError as err:
                        err.cArgs = cArguments
                        err.pyArgs = pyArgs
                        raise err
                    storeValues(result, self, pyArgs, cArguments)
                    return result
                return wrapperCall
        elif returnValues:

            def wrapperCall(*args):
                """Wrapper with all save storeValues"""
                pyArgs = tuple(calculate_pyArgs(args))
                cArguments = pyArgs
                try:
                    result = wrappedOperation(*cArguments)
                except ctypes.ArgumentError as err:
                    err.args = err.args + (cArguments,)
                    raise err
                except error.GLError as err:
                    err.cArgs = cArguments
                    err.pyArgs = pyArgs
                    raise err
                return returnValues(result, self, pyArgs, cArguments)
            return wrapperCall
        else:

            def wrapperCall(*args):
                """Wrapper with all save returnValues and storeValues"""
                pyArgs = tuple(calculate_pyArgs(args))
                cArguments = pyArgs
                try:
                    result = wrappedOperation(*cArguments)
                except ctypes.ArgumentError as err:
                    err.args = err.args + (cArguments,)
                    raise err
                except error.GLError as err:
                    err.cArgs = cArguments
                    err.pyArgs = pyArgs
                    raise err
                return result
            return wrapperCall
    elif cConverters:
        if cResolvers:
            if storeValues:
                if returnValues:

                    def wrapperCall(*args):
                        """Wrapper with all possible operations"""
                        pyArgs = args
                        cArgs = []
                        for index, converter in enumerate(cConverters):
                            if not hasattr(converter, '__call__'):
                                cArgs.append(converter)
                            else:
                                try:
                                    cArgs.append(converter(pyArgs, index, self))
                                except Exception as err:
                                    if hasattr(err, 'args'):
                                        err.args += ('Failure in cConverter %r' % converter, pyArgs, index)
                                    raise
                        cArguments = tuple(calculate_cArguments(cArgs))
                        try:
                            result = wrappedOperation(*cArguments)
                        except ctypes.ArgumentError as err:
                            err.args = err.args + (cArguments,)
                            raise err
                        except error.GLError as err:
                            err.cArgs = cArgs
                            err.pyArgs = pyArgs
                            raise err
                        storeValues(result, self, pyArgs, cArgs)
                        return returnValues(result, self, pyArgs, cArgs)
                    return wrapperCall
                else:

                    def wrapperCall(*args):
                        """Wrapper with all save returnValues"""
                        pyArgs = args
                        cArgs = []
                        for index, converter in enumerate(cConverters):
                            if not hasattr(converter, '__call__'):
                                cArgs.append(converter)
                            else:
                                try:
                                    cArgs.append(converter(pyArgs, index, self))
                                except Exception as err:
                                    if hasattr(err, 'args'):
                                        err.args += ('Failure in cConverter %r' % converter, pyArgs, index)
                                    raise
                        cArguments = tuple(calculate_cArguments(cArgs))
                        try:
                            result = wrappedOperation(*cArguments)
                        except ctypes.ArgumentError as err:
                            err.args = err.args + (cArguments,)
                            raise err
                        except error.GLError as err:
                            err.cArgs = cArgs
                            err.pyArgs = pyArgs
                            raise err
                        storeValues(result, self, pyArgs, cArgs)
                        return result
                    return wrapperCall
            elif returnValues:

                def wrapperCall(*args):
                    """Wrapper with all save storeValues"""
                    pyArgs = args
                    cArgs = []
                    for index, converter in enumerate(cConverters):
                        if not hasattr(converter, '__call__'):
                            cArgs.append(converter)
                        else:
                            try:
                                cArgs.append(converter(pyArgs, index, self))
                            except Exception as err:
                                if hasattr(err, 'args'):
                                    err.args += ('Failure in cConverter %r' % converter, pyArgs, index)
                                raise
                    cArguments = tuple(calculate_cArguments(cArgs))
                    try:
                        result = wrappedOperation(*cArguments)
                    except ctypes.ArgumentError as err:
                        err.args = err.args + (cArguments,)
                        raise err
                    except error.GLError as err:
                        err.cArgs = cArgs
                        err.pyArgs = pyArgs
                        raise err
                    return returnValues(result, self, pyArgs, cArgs)
                return wrapperCall
            else:

                def wrapperCall(*args):
                    """Wrapper with all save returnValues and storeValues"""
                    pyArgs = args
                    cArgs = []
                    for index, converter in enumerate(cConverters):
                        if not hasattr(converter, '__call__'):
                            cArgs.append(converter)
                        else:
                            try:
                                cArgs.append(converter(pyArgs, index, self))
                            except Exception as err:
                                if hasattr(err, 'args'):
                                    err.args += ('Failure in cConverter %r' % converter, pyArgs, index)
                                raise
                    cArguments = tuple(calculate_cArguments(cArgs))
                    try:
                        result = wrappedOperation(*cArguments)
                    except ctypes.ArgumentError as err:
                        err.args = err.args + (cArguments,)
                        raise err
                    except error.GLError as err:
                        err.cArgs = cArgs
                        err.pyArgs = pyArgs
                        raise err
                    return result
                return wrapperCall
        elif storeValues:
            if returnValues:

                def wrapperCall(*args):
                    """Wrapper with all possible operations"""
                    pyArgs = args
                    cArgs = []
                    for index, converter in enumerate(cConverters):
                        if not hasattr(converter, '__call__'):
                            cArgs.append(converter)
                        else:
                            try:
                                cArgs.append(converter(pyArgs, index, self))
                            except Exception as err:
                                if hasattr(err, 'args'):
                                    err.args += ('Failure in cConverter %r' % converter, pyArgs, index)
                                raise
                    cArguments = cArgs
                    try:
                        result = wrappedOperation(*cArguments)
                    except ctypes.ArgumentError as err:
                        err.args = err.args + (cArguments,)
                        raise err
                    except error.GLError as err:
                        err.cArgs = cArgs
                        err.pyArgs = pyArgs
                        raise err
                    storeValues(result, self, pyArgs, cArgs)
                    return returnValues(result, self, pyArgs, cArgs)
                return wrapperCall
            else:

                def wrapperCall(*args):
                    """Wrapper with all save returnValues"""
                    pyArgs = args
                    cArgs = []
                    for index, converter in enumerate(cConverters):
                        if not hasattr(converter, '__call__'):
                            cArgs.append(converter)
                        else:
                            try:
                                cArgs.append(converter(pyArgs, index, self))
                            except Exception as err:
                                if hasattr(err, 'args'):
                                    err.args += ('Failure in cConverter %r' % converter, pyArgs, index)
                                raise
                    cArguments = cArgs
                    try:
                        result = wrappedOperation(*cArguments)
                    except ctypes.ArgumentError as err:
                        err.args = err.args + (cArguments,)
                        raise err
                    except error.GLError as err:
                        err.cArgs = cArgs
                        err.pyArgs = pyArgs
                        raise err
                    storeValues(result, self, pyArgs, cArgs)
                    return result
                return wrapperCall
        elif returnValues:

            def wrapperCall(*args):
                """Wrapper with all save storeValues"""
                pyArgs = args
                cArgs = []
                for index, converter in enumerate(cConverters):
                    if not hasattr(converter, '__call__'):
                        cArgs.append(converter)
                    else:
                        try:
                            cArgs.append(converter(pyArgs, index, self))
                        except Exception as err:
                            if hasattr(err, 'args'):
                                err.args += ('Failure in cConverter %r' % converter, pyArgs, index)
                            raise
                cArguments = cArgs
                try:
                    result = wrappedOperation(*cArguments)
                except ctypes.ArgumentError as err:
                    err.args = err.args + (cArguments,)
                    raise err
                except error.GLError as err:
                    err.cArgs = cArgs
                    err.pyArgs = pyArgs
                    raise err
                return returnValues(result, self, pyArgs, cArgs)
            return wrapperCall
        else:

            def wrapperCall(*args):
                """Wrapper with all save returnValues and storeValues"""
                pyArgs = args
                cArgs = []
                for index, converter in enumerate(cConverters):
                    if not hasattr(converter, '__call__'):
                        cArgs.append(converter)
                    else:
                        try:
                            cArgs.append(converter(pyArgs, index, self))
                        except Exception as err:
                            if hasattr(err, 'args'):
                                err.args += ('Failure in cConverter %r' % converter, pyArgs, index)
                            raise
                cArguments = cArgs
                try:
                    result = wrappedOperation(*cArguments)
                except ctypes.ArgumentError as err:
                    err.args = err.args + (cArguments,)
                    raise err
                except error.GLError as err:
                    err.cArgs = cArgs
                    err.pyArgs = pyArgs
                    raise err
                return result
            return wrapperCall
    elif cResolvers:
        if storeValues:
            if returnValues:

                def wrapperCall(*args):
                    """Wrapper with all possible operations"""
                    cArgs = args
                    cArguments = tuple(calculate_cArguments(cArgs))
                    try:
                        result = wrappedOperation(*cArguments)
                    except ctypes.ArgumentError as err:
                        err.args = err.args + (cArguments,)
                        raise err
                    except error.GLError as err:
                        err.cArgs = cArgs
                        err.pyArgs = args
                        raise err
                    storeValues(result, self, args, cArgs)
                    return returnValues(result, self, args, cArgs)
                return wrapperCall
            else:

                def wrapperCall(*args):
                    """Wrapper with all save returnValues"""
                    cArgs = args
                    cArguments = tuple(calculate_cArguments(cArgs))
                    try:
                        result = wrappedOperation(*cArguments)
                    except ctypes.ArgumentError as err:
                        err.args = err.args + (cArguments,)
                        raise err
                    except error.GLError as err:
                        err.cArgs = cArgs
                        err.pyArgs = args
                        raise err
                    storeValues(result, self, args, cArgs)
                    return result
                return wrapperCall
        elif returnValues:

            def wrapperCall(*args):
                """Wrapper with all save storeValues"""
                cArgs = args
                cArguments = tuple(calculate_cArguments(cArgs))
                try:
                    result = wrappedOperation(*cArguments)
                except ctypes.ArgumentError as err:
                    err.args = err.args + (cArguments,)
                    raise err
                except error.GLError as err:
                    err.cArgs = cArgs
                    err.pyArgs = args
                    raise err
                return returnValues(result, self, args, cArgs)
            return wrapperCall
        else:

            def wrapperCall(*args):
                """Wrapper with all save returnValues and storeValues"""
                cArgs = args
                cArguments = tuple(calculate_cArguments(cArgs))
                try:
                    result = wrappedOperation(*cArguments)
                except ctypes.ArgumentError as err:
                    err.args = err.args + (cArguments,)
                    raise err
                except error.GLError as err:
                    err.cArgs = cArgs
                    err.pyArgs = args
                    raise err
                return result
            return wrapperCall
    elif storeValues:
        if returnValues:

            def wrapperCall(*args):
                """Wrapper with all possible operations"""
                cArguments = args
                try:
                    result = wrappedOperation(*cArguments)
                except ctypes.ArgumentError as err:
                    err.args = err.args + (cArguments,)
                    raise err
                except error.GLError as err:
                    err.cArgs = cArguments
                    err.pyArgs = args
                    raise err
                storeValues(result, self, args, cArguments)
                return returnValues(result, self, args, cArguments)
            return wrapperCall
        else:

            def wrapperCall(*args):
                """Wrapper with all save returnValues"""
                cArguments = args
                try:
                    result = wrappedOperation(*cArguments)
                except ctypes.ArgumentError as err:
                    err.args = err.args + (cArguments,)
                    raise err
                except error.GLError as err:
                    err.cArgs = cArguments
                    err.pyArgs = args
                    raise err
                storeValues(result, self, args, cArguments)
                return result
            return wrapperCall
    elif returnValues:

        def wrapperCall(*args):
            """Wrapper with all save storeValues"""
            cArguments = args
            try:
                result = wrappedOperation(*cArguments)
            except ctypes.ArgumentError as err:
                err.args = err.args + (cArguments,)
                raise err
            except error.GLError as err:
                err.cArgs = cArguments
                err.pyArgs = args
                raise err
            return returnValues(result, self, args, cArguments)
        return wrapperCall
    else:

        def wrapperCall(*args):
            """Wrapper with all save returnValues and storeValues"""
            cArguments = args
            try:
                result = wrappedOperation(*cArguments)
            except ctypes.ArgumentError as err:
                err.args = err.args + (cArguments,)
                raise err
            except error.GLError as err:
                err.cArgs = cArguments
                err.pyArgs = args
                raise err
            return result
        return wrapperCall