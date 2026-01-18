import dataclasses
from typing import cast, Optional, Sequence, Union
import cirq
from cirq_google import ProcessorSampler, get_engine
from cirq_google.engine import (
def get_qcs_objects_for_notebook(project_id: Optional[str]=None, processor_id: Optional[str]=None, virtual=False) -> QCSObjectsForNotebook:
    """Authenticates on Google Cloud and returns Engine related objects.

    This function will authenticate to Google Cloud and attempt to
    instantiate an Engine object.  If it does not succeed, it will instead
    return a virtual AbstractEngine that is backed by a noisy simulator.
    This function is designed for maximum versatility and
    to work in colab notebooks, as a stand-alone, and in tests.

    Note that, if you are using this to connect to QCS and do not care about
    the added versatility, you may want to use `cirq_google.get_engine()` or
    `cirq_google.Engine()` instead to guarantee the use of a production instance
    and to avoid accidental use of a noisy simulator.

    Args:
        project_id: Optional explicit Google Cloud project id. Otherwise,
            this defaults to the environment variable GOOGLE_CLOUD_PROJECT.
            By using an environment variable, you can avoid hard-coding
            personal project IDs in shared code.
        processor_id: Engine processor ID (from Cloud console or
            ``Engine.list_processors``).
        virtual: If set to True, will create a noisy virtual Engine instead.
            This is useful for testing and simulation.

    Returns:
        An instance of QCSObjectsForNotebook which contains all the objects .

    Raises:
        ValueError: if processor_id is not specified and no processors are available.
    """
    if not virtual:
        try:
            from google.colab import auth
        except ImportError:
            print('Not running in a colab kernel. Will use Application Default Credentials.')
        else:
            print('Getting OAuth2 credentials.')
            print('Press enter after entering the verification code.')
            try:
                a = auth.authenticate_user(clear_output=False)
                print(a)
                print('Authentication complete.')
            except Exception as exc:
                print(f'Authentication failed: {exc}')
                print('Using virtual engine instead.')
                virtual = True
    if not virtual:
        try:
            engine: AbstractEngine = get_engine(project_id)
            signed_in = True
            is_simulator = False
        except Exception as exc:
            print(f'Unable to connect to quantum engine: {exc}')
            print('Using a noisy simulator.')
            virtual = True
    if virtual:
        engine = create_noiseless_virtual_engine_from_latest_templates()
        signed_in = False
        is_simulator = True
    if processor_id:
        processor = engine.get_processor(processor_id)
    else:
        processors = cast(Sequence[Union[EngineProcessor, AbstractLocalProcessor]], engine.list_processors())
        if not processors:
            raise ValueError('No processors available.')
        processor = processors[0]
        processor_id = processor.processor_id
        print(f'Available processors: {[p.processor_id for p in processors]}')
        print(f'Using processor: {processor_id}')
    if not project_id:
        project_id = getattr(processor, 'project_id', None)
    device = processor.get_device()
    sampler = processor.get_sampler()
    return QCSObjectsForNotebook(engine=engine, processor=processor, device=device, sampler=sampler, signed_in=signed_in, project_id=project_id, processor_id=processor_id, is_simulator=is_simulator)